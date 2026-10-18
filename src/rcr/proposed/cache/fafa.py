"""Independent frozen FAFA image and content-addressed query extraction."""

import pickle
import shutil
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from rcr.baselines.fafa import load_fafa, text_policy
from rcr.common.data import load_rcr_data
from rcr.common.io import sha256_file
from rcr.common.runtime import resolve_device
from rcr.common.torch_io import require_disk_space, save_torch
from rcr.proposed.cache.build import (
    cache_paths,
    check_cache_config,
    existing_cache,
    fafa_spec,
    load_fafa_config,
    query_cache_ready,
)
from rcr.proposed.cache.native import (
    NATIVE_VERSION,
    cached_text_policy,
    condition_key,
    query_entries,
    subject_conditions,
)
from rcr.proposed.cache.store import GalleryCache


def boxes_to_pixels(
    boxes: torch.Tensor,
    image_size: tuple[int, int],
    scene_size: tuple[int, int],
) -> torch.Tensor:
    """Invert the legacy ImageOps.pad transform without rerunning detection."""
    width, height = image_size
    scene_width, scene_height = scene_size
    scale = min(scene_width / width, scene_height / height)
    resized_width = min(scene_width, round(width * scale))
    resized_height = min(scene_height, round(height * scale))
    left = round((scene_width - resized_width) * 0.5)
    top = round((scene_height - resized_height) * 0.5)
    size = boxes.new_tensor([scene_width, scene_height] * 2)
    offset = boxes.new_tensor([left, top, left, top])
    factors = boxes.new_tensor([resized_width / width, resized_height / height] * 2)
    return (
        ((boxes * size - offset) / factors)
        .clamp(min=0)
        .minimum(boxes.new_tensor([width, height] * 2))
    )


def _ready(path, cache_id, key, shape, dtype):
    if not path.is_file():
        return False
    try:
        item = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):
        return False
    features = item.get("features") if isinstance(item, dict) else None
    return (
        isinstance(item, dict)
        and item.get("cache_id") == cache_id
        and item.get("key") == key
        and isinstance(features, torch.Tensor)
        and features.shape == shape
        and features.dtype == dtype
        and bool(torch.isfinite(features).all())
    )


def _begin(root, signature, *, cache_id=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".build.pt"
    if path.is_file():
        manifest = torch.load(path, map_location="cpu", weights_only=True)
        if manifest["signature"] != signature or (
            cache_id is not None and manifest["cache_id"] != cache_id
        ):
            raise ValueError(
                "unfinished FAFA cache settings differ; use a new directory"
            )
    else:
        manifest = {"cache_id": cache_id or uuid4().hex, "signature": signature}
        save_torch(manifest, path)
    (root / ".building").write_text(manifest["cache_id"], encoding="utf-8")
    return manifest["cache_id"]


def _publish(root, index):
    save_torch(index, root / "index.pt")
    (root / ".building").unlink()
    (root / ".build.pt").unlink(missing_ok=True)


def _write(path, cache_id, key, features):
    if not torch.isfinite(features).all():
        raise ValueError(f"non-finite FAFA features: {path}")
    save_torch({"cache_id": cache_id, "key": key, "features": features}, path)


class _Extractor:
    """One lazy model load shared by both stages in the isolated FAFA process."""

    def __init__(self, cfg, source, data):
        self.cfg, self.source, self.data = cfg, source, data
        self.batch_size = cfg["person_encoder"]["batch_size"]
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("person_encoder.batch_size must be positive")
        self.native_cfg = load_fafa_config(cfg)
        self.text_policy = text_policy(self.native_cfg)
        self.device = resolve_device(cfg)
        self.signature = {
            "source_cache_id": source.cache_id,
            "spec": fafa_spec(self.native_cfg),
            "checkpoint_sha256": sha256_file(self.native_cfg["checkpoint"]["path"]),
            "image_encoder": cfg["image_encoder"],
            "detector": cfg["detector"],
            "encoder_precision": "float16" if self.device.type == "cuda" else "float32",
        }
        self.model = None

    def load(self):
        if self.model is None:
            model, self.processors, self.preprocess, _ = load_fafa(
                self.native_cfg, self.device
            )
            self.model = model.requires_grad_(False).eval()
            self.hidden_dim = model.Qformer.config.hidden_size
            self.dim = model.vision_proj.out_features
            self.tokens = model.query_tokens.shape[1]

    def crops(self, image_id):
        i = self.source.by_id[image_id]
        count = int(self.source.mask[i].sum())
        if count == 0:
            return
        item = self.source._load_item(i)
        with Image.open(self.data.image_path(image_id)) as image_file:
            image = image_file.convert("RGB")
            boxes = item.get("boxes_pixel")
            if boxes is None:
                h, w = self.cfg["image_encoder"]["scene_size"]
                boxes = boxes_to_pixels(item["boxes_scene"], image.size, (w, h))
            if (
                boxes.shape != (count, 4)
                or not torch.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
            ):
                raise ValueError("invalid DINO person boxes")
            for start in range(0, count, self.batch_size):
                pixels = torch.stack(
                    [
                        self.preprocess(image.crop(tuple(box.tolist())))
                        for box in boxes[start : start + self.batch_size]
                    ]
                ).to(self.device)
                yield (
                    start,
                    pixels.half() if self.device.type == "cuda" else pixels.float(),
                )


def _build_images(cfg, root, extractor):
    storage = cfg.get("cache", {}).get("storage_dtype", "float16")
    if storage not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float32 or float16")
    dtype = getattr(torch, storage)
    signature = {**extractor.signature, "storage_dtype": storage}
    cache_id = _begin(root, signature)
    for group in ("targets", ".persons"):
        (root / group).mkdir(exist_ok=True)
    extractor.load()
    source = extractor.source
    hidden_dim, dim, tokens = extractor.hidden_dim, extractor.dim, extractor.tokens
    counts = source.mask.sum(dim=1)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    total = int(offsets[-1])
    pooled_bytes = total * hidden_dim * dtype.itemsize
    target_bytes = total * tokens * dim * dtype.itemsize
    # The compact index coexists with resumable pooled shards until publication.
    # DINO owns the global scene features; no second copy is written here.
    estimated = (
        target_bytes
        + 2 * pooled_bytes
        + source.mask.numel()
        + offsets.numel() * offsets.element_size()
        + (2 * len(source.image_ids) + 1) * 4096
        + 2 * len(str(source.image_ids).encode())
    )
    complete = set()
    saved_bytes = 0
    for i, image_id in enumerate(tqdm(source.image_ids, desc="check FAFA images")):
        count = int(counts[i])
        person_path, target_path = (
            root / group / f"{i}.pt" for group in (".persons", "targets")
        )
        if _ready(
            person_path, cache_id, image_id, (count, hidden_dim), dtype
        ) and _ready(target_path, cache_id, image_id, (count, tokens, dim), dtype):
            complete.add(i)
            # Only valid reusable shards count toward the reservation. Cap the
            # credit so oversized storages/stale files cannot bypass the guard.
            saved_bytes += min(
                person_path.stat().st_size, count * hidden_dim * dtype.itemsize + 4096
            ) + min(
                target_path.stat().st_size,
                count * tokens * dim * dtype.itemsize + 4096,
            )
    padded_bytes = source.mask.numel() * hidden_dim * dtype.itemsize
    print(
        f"FAFA images: {len(complete)}/{len(source.image_ids)} ready, {total} persons; "
        f"person index {padded_bytes / 1024**3:.2f} -> "
        f"{pooled_bytes / 1024**3:.2f} GiB (packed); "
        f"target tokens {target_bytes / 1024**3:.2f} GiB; "
        f"peak estimate {estimated / 1024**3:.2f} GiB",
        flush=True,
    )
    require_disk_space(root, max(0, estimated - saved_bytes))
    for i, image_id in enumerate(tqdm(source.image_ids, desc="cache FAFA images")):
        if i in complete:
            continue
        count = int(counts[i])
        person_path, target_path = (
            root / group / f"{i}.pt" for group in (".persons", "targets")
        )
        persons = torch.empty(count, hidden_dim, dtype=dtype)
        targets = torch.empty(count, tokens, dim, dtype=dtype)
        for start, pixels in extractor.crops(image_id):
            value = extractor.model.extract_features({"image": pixels}, mode="image")
            hidden, projected = value.image_embeds, value.image_embeds_proj
            if hidden.shape != (len(pixels), tokens, hidden_dim):
                raise ValueError("FAFA image_embeds must be [N,L,hidden_dim]")
            if projected.shape != (len(pixels), tokens, dim):
                raise ValueError("FAFA projected tokens must be [N,L,D_fafa]")
            stop = start + len(pixels)
            persons[start:stop] = hidden.float().mean(1).cpu().to(dtype)
            targets[start:stop] = projected.float().cpu().to(dtype)
        _write(person_path, cache_id, image_id, persons)
        _write(target_path, cache_id, image_id, targets)
    persons_index = torch.empty(total, hidden_dim, dtype=dtype)
    for i in range(len(source.image_ids)):
        item = torch.load(
            root / ".persons" / f"{i}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        persons_index[int(offsets[i]) : int(offsets[i + 1])] = item["features"]
    _publish(
        root,
        {
            "format_version": 6,
            "layout": "fafa",
            "cache_id": cache_id,
            "source_cache_id": source.cache_id,
            "signature": extractor.signature,
            "image_ids": source.image_ids,
            "persons": persons_index,
            "person_offsets": offsets,
            "mask": source.mask,
            "patch_hw": source.patch_hw,
            "scene_dim": source.scene_dim,
            "storage_dtype": storage,
            "image_encoder": cfg["image_encoder"],
            "detector": cfg["detector"],
            "person_encoder": {
                "backend": "fafa",
                "spec": signature["spec"],
                "checkpoint_sha256": signature["checkpoint_sha256"],
                "hidden_dim": hidden_dim,
                "encoder_precision": signature["encoder_precision"],
            },
            "native": {"version": NATIVE_VERSION, "dim": dim, "tokens": tokens},
        },
    )
    shutil.rmtree(root / ".persons")
    print(f"Finished FAFA image cache: {root}", flush=True)


def _query_index(root, signature):
    if not (root / "index.pt").is_file():
        return None
    index = torch.load(root / "index.pt", map_location="cpu", weights_only=True)
    if (
        index.get("layout") != "fafa-queries"
        or index.get("version") != NATIVE_VERSION
        or index.get("signature") != signature
    ):
        raise ValueError("FAFA query encoder/crops differ; use a new query directory")
    cached_text_policy(index)
    return index


def _build_queries(root, extractor):
    signature, source = extractor.signature, extractor.source
    policy = extractor.text_policy
    entries = query_entries(extractor.data.samples, signature, policy)
    previous = _query_index(root, signature)
    same_policy = previous is not None and cached_text_policy(previous) == policy
    if (
        previous is not None
        and same_policy
        and not (root / ".building").exists()
        and previous["queries"] == entries
    ):
        print(f"Using existing FAFA query cache: {root}", flush=True)
        return
    # Preserve legacy resume signatures. New limits belong to query extraction
    # only, so completed image indices and target shards never change.
    build_signature = (
        signature
        if policy == text_policy({})
        else {"encoder": signature, "text_policy": policy}
    )
    cache_id = _begin(
        root, build_signature, cache_id=previous["cache_id"] if same_policy else None
    )
    (root / "features").mkdir(exist_ok=True)
    if previous is None:
        extractor.load()
    dim = previous["dim"] if previous is not None else extractor.dim
    work = {}
    for row in extractor.data.samples:
        image_id = row["query_image_id"]
        count = int(source.mask[source.by_id[image_id]].sum())
        for condition in subject_conditions(row):
            key = condition_key(signature, image_id, condition, policy)
            work[key] = (image_id, condition, count)
    missing = {
        key: value
        for key, value in work.items()
        if not _ready(
            root / "features" / f"{key}.pt",
            cache_id,
            key,
            (value[2], dim),
            torch.float32,
        )
    }
    estimated = sum(count * dim * 4 + 4096 for _, _, count in missing.values())
    # Atomic replacement temporarily retains the prior manifest.
    estimated += len(str(entries).encode()) * 2
    require_disk_space(root, estimated)
    print(
        f"FAFA queries: {len(missing)}/{len(work)} conditions to extract, "
        f"{estimated / 1024**3:.2f} GiB additional",
        flush=True,
    )
    if missing:
        extractor.load()
        if extractor.dim != dim:
            raise ValueError("FAFA query feature dimension differs from cache")
    stats = {"conditions_checked": 0, "over_token_limit": 0, "max_tokens": 0}
    for key, (image_id, condition, count) in tqdm(
        missing.items(), desc="cache FAFA queries"
    ):
        processed = extractor.processors["eval"](condition)
        tokens = extractor.model.tokenizer(processed, truncation=False)["input_ids"]
        length = len(tokens)
        stats["conditions_checked"] += 1
        stats["over_token_limit"] += int(length > policy["max_txt_len"])
        stats["max_tokens"] = max(stats["max_tokens"], length)
        features = torch.empty(count, dim)
        for start, pixels in extractor.crops(image_id):
            value = extractor.model.extract_features(
                {
                    "image": pixels,
                    "text_input": [processed] * len(pixels),
                },
                mode="multimodal",
            ).multimodal_embeds.float()
            if value.ndim == 3 and value.shape[1] == 1:
                value = value[:, 0]
            if value.shape != (len(pixels), dim):
                raise ValueError("FAFA multimodal query must be [N,D_fafa]")
            features[start : start + len(pixels)] = value.cpu()
        _write(root / "features" / f"{key}.pt", cache_id, key, features)
    _publish(
        root,
        {
            "layout": "fafa-queries",
            "version": NATIVE_VERSION,
            "cache_id": cache_id,
            "signature": signature,
            "text_policy": policy,
            # Counts cover conditions extracted by this invocation, not reused shards.
            "text_stats": stats,
            "dim": dim,
            "queries": entries,
        },
    )
    print(
        f"FAFA query text: max_txt_len={policy['max_txt_len']}, "
        f"max_words={policy['max_words']}; "
        f"{stats['over_token_limit']}/{stats['conditions_checked']} new conditions "
        f"exceed the token limit (max_tokens={stats['max_tokens']})",
        flush=True,
    )
    print(f"Finished FAFA query cache: {root}", flush=True)


@torch.inference_mode()
def finish_fafa_cache(cfg: dict, *, stage: str = "fafa") -> None:
    if stage not in ("fafa", "fafa-image", "fafa-query"):
        raise ValueError("invalid FAFA cache stage")
    source_root, image_root, query_root = cache_paths(cfg)
    source = GalleryCache(source_root)
    check_cache_config(source, cfg, person_backend="dino")
    if source.cache_id is None:
        raise ValueError("DINO source requires a cache_id to bind FAFA features safely")
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    source.validate_gallery(data.gallery_ids)
    image = existing_cache(cfg, image_root, scene_root=source_root)
    if image is not None:
        image.validate_gallery(data.gallery_ids)
        if stage == "fafa-image":
            print(f"Using existing FAFA image cache: {image_root}", flush=True)
            return
        if query_cache_ready(
            query_root, image, data.samples, text_policy(load_fafa_config(cfg))
        ):
            print(f"Using existing FAFA query cache: {query_root}", flush=True)
            return
    extractor = _Extractor(cfg, source, data)
    if image is not None and image.feature_signature != extractor.signature:
        raise ValueError("FAFA query encoder/precision differs from image cache")
    if stage != "fafa-image":
        _query_index(query_root, extractor.signature)
    if stage != "fafa-query" and image is None:
        _build_images(cfg, image_root, extractor)
    if stage != "fafa-image":
        _build_queries(query_root, extractor)
