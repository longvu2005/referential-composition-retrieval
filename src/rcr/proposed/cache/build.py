"""Select cache stages and run the isolated FAFA worker."""

import subprocess
import tempfile
from pathlib import Path

import yaml

from rcr.common.config import load_config
from rcr.common.data import load_rcr_data
from rcr.common.io import sha256_file
from rcr.proposed.cache.dino import _prepare_dino
from rcr.proposed.cache.native import NativeQueryCache
from rcr.proposed.cache.store import GalleryCache


def load_fafa_config(cfg: dict) -> dict:
    path = Path(cfg["person_encoder"]["fafa_config"])
    return load_config(path)


def fafa_spec(cfg: dict) -> dict:
    """Feature provenance independent of baseline detector/selector settings."""
    return {
        "source_commit": cfg["source"]["commit"],
        "checkpoint_source": cfg["checkpoint"].get("source_url"),
        "model_name": cfg["checkpoint"]["model_name"],
        "model_type": cfg["checkpoint"]["model_type"],
        "image_size": cfg["checkpoint"]["image_size"],
        "test_resize_hw": cfg["checkpoint"]["test_resize_hw"],
        "feature": "pooled_image+projected_tokens+multimodal-v1",
    }


def run_person_worker(
    cfg: dict, *, stage: str = "fafa", prepare: bool = False, force: bool = False
) -> None:
    """Share files only; native FAFA stays in its own dependency environment."""
    worker = Path(__file__).resolve().parents[4] / "tools/cache_fafa.py"
    with tempfile.TemporaryDirectory(prefix="rcr-person-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [cfg["person_encoder"]["python"], str(worker), "--config", str(path)]
        if not prepare:
            command.extend(["--stage", stage])
        if prepare:
            command.append("--prepare")
        if force:
            command.append("--force")
        subprocess.run(command, check=True)


def check_cache_config(
    cache: GalleryCache, cfg: dict, *, person_backend: str = "fafa"
) -> None:
    """Check encoder provenance only when building or explicitly reusing a cache."""
    if "person_encoder" not in cfg:
        return  # Small in-memory test configs have no external encoder spec.
    if cache._scene_cache is not None:
        check_cache_config(cache._scene_cache, cfg, person_backend="dino")
    if cache.format_version == 1:
        if person_backend != "dino" or not cfg.get("cache", {}).get(
            "allow_legacy_dino", False
        ):
            raise ValueError("legacy DINO cache requires cache.allow_legacy_dino=true")
        # Legacy files cannot prove their model/detector provenance. This is
        # an explicit user assertion, never inferred from equal feature widths.
        height, width = cfg["image_encoder"]["scene_size"]
        if cache.patch_hw != (height // 16, width // 16):
            raise ValueError("legacy DINO patch grid differs from scene_size")
        return
    if cache.format_version not in (2, 4, 5, 6):
        raise ValueError("unsupported cache format")
    actual = cache.encoder_metadata["person_encoder"] or {}
    if actual.get("backend") != person_backend:
        raise ValueError("cache person encoder differs; use a matching cache directory")
    if cache.encoder_metadata["image_encoder"] != cfg["image_encoder"]:
        raise ValueError("cache image encoder/preprocessing differs from config")
    if cache.encoder_metadata["detector"] != cfg["detector"]:
        raise ValueError("cache detector differs from config")
    if person_backend == "fafa":
        if cache.format_version not in (4, 5, 6) or cache.native is None:
            raise ValueError("build a FAFA image cache in data.fafa_image_cache")
        native_cfg = load_fafa_config(cfg)
        if actual.get("spec") != fafa_spec(native_cfg):
            raise ValueError("cache FAFA checkpoint/source/preprocessing spec differs")
        checkpoint = Path(native_cfg["checkpoint"]["path"])
        if checkpoint.is_file():
            if actual.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError("cache FAFA checkpoint weights differ")


def existing_cache(
    cfg: dict,
    root: Path,
    *,
    scene_root: Path | None = None,
    person_backend: str = "fafa",
):
    if not (root / "index.pt").is_file() or (root / ".building").exists():
        return None
    cache = GalleryCache(root, scene_root=scene_root)
    check_cache_config(cache, cfg, person_backend=person_backend)
    return cache


def query_cache_ready(root: Path, image: GalleryCache, samples) -> bool:
    if not (root / "index.pt").is_file() or (root / ".building").exists():
        return False
    queries = NativeQueryCache(root, image.feature_signature, image.native.dim)
    return queries.contains_samples(samples)


def cache_paths(cfg: dict) -> tuple[Path, Path, Path]:
    """Keep every cache separate, including through symlinks and parent paths."""
    data = cfg["data"]
    if "cache" in data or "native_cache" in data:
        raise ValueError(
            "replace data.cache/data.native_cache with data.fafa_image_cache and "
            "data.fafa_query_cache; a completed unified cache can be reused as "
            "fafa_image_cache"
        )
    paths = tuple(
        Path(data[key])
        for key in ("dino_cache", "fafa_image_cache", "fafa_query_cache")
    )
    resolved = [path.resolve() for path in paths]
    for i, first in enumerate(resolved):
        for second in resolved[i + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError(
                    "DINO, FAFA image and query caches must be separate directories"
                )
    return paths


def prepare_cache(cfg: dict, *, stage: str = "all") -> None:
    """Build each cache independently; never write to a completed image cache."""
    if stage not in ("all", "dino", "fafa", "fafa-image", "fafa-query"):
        raise ValueError("invalid cache stage")
    root, image_root, query_root = cache_paths(cfg)
    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    source = existing_cache(cfg, root, person_backend="dino")
    if source is not None:
        source.validate_gallery(data.gallery_ids)
        print(f"Using existing DINO cache: {root}", flush=True)
    elif stage not in ("all", "dino"):
        raise ValueError("DINO cache missing/incomplete; run --cache-stage dino first")
    if source is None:
        _prepare_dino(cfg, data, root)
    if stage == "dino":
        return
    image = existing_cache(cfg, image_root, scene_root=root)
    if image is not None:
        image.validate_gallery(data.gallery_ids)
    if stage == "fafa-image" and image is not None:
        print(f"Using existing FAFA image cache: {image_root}", flush=True)
        return
    if (
        stage != "fafa-image"
        and image is not None
        and query_cache_ready(query_root, image, data.samples)
    ):
        print(f"Using existing FAFA caches: {image_root}, {query_root}", flush=True)
        return
    run_person_worker(cfg, stage="fafa" if stage == "all" else stage)
    if stage != "fafa-query":
        image = existing_cache(cfg, image_root, scene_root=root)
        if image is None:
            raise RuntimeError("FAFA image build did not publish index.pt")
    if stage != "fafa-image":
        if (
            not (query_root / "index.pt").is_file()
            or (query_root / ".building").exists()
        ):
            raise RuntimeError("FAFA query build did not publish index.pt")
        # Query-only extraction also works before the image cache is built.
        if image is not None:
            NativeQueryCache(
                query_root, image.feature_signature, image.native.dim
            ).validate_samples(data.samples)
