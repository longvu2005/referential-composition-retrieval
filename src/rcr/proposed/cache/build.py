"""Select cache stages; the FAFA worker uses the current Python interpreter."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from rcr.common.config import load_config
from rcr.common.data import load_rcr_data
from rcr.common.io import sha256_file
from rcr.common.storage import format_bytes
from rcr.proposed.cache.dino import _prepare_dino
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
        "feature": "mean_image_embeds_before_vision_proj",
    }


def run_person_worker(cfg: dict, *, prepare: bool = False, force: bool = False) -> None:
    """Use this runtime in a fresh process to release FAFA before the CLIP stage."""
    if cfg.get("person_encoder", {}).get("backend", "dino") == "dino":
        return
    worker = Path(__file__).resolve().parents[4] / "tools/cache_fafa.py"
    with tempfile.TemporaryDirectory(prefix="rcr-person-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [sys.executable, str(worker), "--config", str(path)]
        if prepare:
            command.append("--prepare")
        if force:
            command.append("--force")
        subprocess.run(command, check=True)


def check_cache_config(cache: GalleryCache, cfg: dict) -> None:
    """Check encoder provenance only when building or explicitly reusing a cache."""
    expected = cfg.get("person_encoder")
    if expected is None:
        return  # Small in-memory test configs have no external encoder spec.
    if cache._scene_cache is not None:
        check_cache_config(
            cache._scene_cache, {**cfg, "person_encoder": {"backend": "dino"}}
        )
    if cache.format_version == 1:
        if expected["backend"] != "dino" or not cfg.get("cache", {}).get(
            "allow_legacy_dino", False
        ):
            raise ValueError("legacy DINO cache requires cache.allow_legacy_dino=true")
        # Legacy files cannot prove their model/detector provenance. This is
        # an explicit user assertion, never inferred from equal feature widths.
        height, width = cfg["image_encoder"]["scene_size"]
        if cache.patch_hw != (height // 16, width // 16):
            raise ValueError("legacy DINO patch grid differs from scene_size")
        return
    if cache.format_version not in (2, 3):
        raise ValueError("unsupported cache format")
    actual = cache.encoder_metadata["person_encoder"] or {}
    if actual.get("backend") != expected["backend"]:
        raise ValueError("cache person encoder differs; use a matching cache directory")
    if cache.encoder_metadata["image_encoder"] != cfg["image_encoder"]:
        raise ValueError("cache image encoder/preprocessing differs from config")
    if cache.encoder_metadata["detector"] != cfg["detector"]:
        raise ValueError("cache detector differs from config")
    if expected["backend"] == "fafa":
        native_cfg = load_fafa_config(cfg)
        if actual.get("spec") != fafa_spec(native_cfg):
            raise ValueError("cache FAFA checkpoint/source/preprocessing spec differs")
        checkpoint = Path(native_cfg["checkpoint"]["path"])
        if checkpoint.is_file():
            if actual.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError("cache FAFA checkpoint weights differ")


def dino_root(cfg: dict) -> Path:
    if cfg.get("person_encoder", {}).get("backend", "dino") == "dino":
        return Path(cfg["data"]["cache"])
    return Path(cfg["data"]["dino_cache"])


def existing_cache(cfg: dict, root: Path, *, scene_root: Path | None = None):
    if not (root / "index.pt").is_file() or (root / ".building").exists():
        return None
    cache = GalleryCache(root, scene_root=scene_root)
    check_cache_config(cache, cfg)
    return cache


def completed_person_cache(cfg: dict):
    """Return a compatible complete FAFA cache, including its DINO binding."""
    if cfg["person_encoder"]["backend"] != "fafa":
        return None
    scene_root = dino_root(cfg)
    if existing_cache(
        {**cfg, "person_encoder": {"backend": "dino"}}, scene_root
    ) is None:
        return None
    ready = existing_cache(cfg, Path(cfg["data"]["cache"]), scene_root=scene_root)
    if ready is not None:
        data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
        ready.validate_gallery(data.gallery_ids)
    return ready


def prepare_person_assets(cfg: dict, *, force: bool = False) -> None:
    if not force and completed_person_cache(cfg) is not None:
        print(
            "FAFA features already cached; skipping native asset preparation",
            flush=True,
        )
        return
    run_person_worker(cfg, prepare=True, force=force)


def release_fafa_assets(cfg: dict) -> None:
    """Release only published FAFA base assets after compatible features exist."""
    if completed_person_cache(cfg) is None:
        raise ValueError(
            "cannot release FAFA assets before its feature cache is complete"
        )
    native = load_fafa_config(cfg)
    root = Path(native["checkpoint"]["cache_root"]).resolve()
    marker = Path(native["checkpoint"]["runtime_assets_marker"])
    if not root.is_dir() or not marker.is_file():
        return
    if os.statvfs(root).f_flag & os.ST_RDONLY:
        print(f"Keeping read-only FAFA runtime assets: {root}", flush=True)
        return
    protected = [
        *cfg["data"].values(),
        native["checkpoint"]["path"],
        cfg["checkpoint"],
        cfg["output"]["dir"],
    ]
    if any(
        Path(path).resolve().is_relative_to(root)
        or Path(path).absolute().is_relative_to(root)
        for path in protected
    ):
        raise ValueError("FAFA runtime cache overlaps protected data/cache/checkpoints")
    saved = json.loads(marker.read_text(encoding="utf-8"))
    identity = {
        "source_commit": native["source"]["commit"],
        "model_name": native["checkpoint"]["model_name"],
        "model_type": native["checkpoint"]["model_type"],
    }
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("FAFA runtime asset marker differs; refusing cleanup")
    paths = []
    for item in saved["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid FAFA runtime asset path")
        # Names used by the pinned native FAFA implementation. Retain unrelated
        # models/files, including ones that happened to appear in the marker.
        base_weight = relative.parts[:3] == ("torch", "hub", "checkpoints") and (
            relative.name in ("eva_vit_g.pth", "blip2_pretrained.pth")
        )
        bert_asset = "models--bert-base-uncased" in relative.parts
        if not (base_weight or bert_asset):
            continue
        path = root / relative
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError("FAFA runtime asset parent escapes its cache")
        if path.is_file():
            if path.stat().st_size != item["size"]:
                raise ValueError("FAFA runtime asset changed; refusing cleanup")
            paths.append(path)
        elif path.is_symlink():
            paths.append(path)
    if not paths:
        return
    probe = Path(cfg["data"]["clip_cache"])
    probe.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(probe).free
    for path in paths:
        path.unlink(missing_ok=True)
    marker.unlink()
    after = shutil.disk_usage(probe).free
    print(
        f"Released FAFA base assets: {len(paths)} files; "
        f"CLIP volume free={format_bytes(after)} "
        f"(change={format_bytes(max(0, after - before))})",
        flush=True,
    )


def prepare_cache(cfg: dict, *, stage: str = "all") -> None:
    """Build DINO and/or FAFA independently; completed caches stay immutable."""
    backend = cfg.get("person_encoder", {}).get("backend", "dino")
    if backend not in ("dino", "fafa"):
        raise ValueError("person_encoder.backend must be dino or fafa")
    if stage not in ("all", "dino", "persons", "clip"):
        raise ValueError("cache stage must be all, dino, persons or clip")
    if stage in ("all", "clip") and backend != "fafa":
        raise ValueError("the proposed model requires person_encoder.backend=fafa")
    if stage == "persons" and backend != "fafa":
        raise ValueError("persons stage requires person_encoder.backend=fafa")
    data_cfg = cfg["data"]
    root = dino_root(cfg)
    if backend == "fafa" and root.resolve() == Path(data_cfg["cache"]).resolve():
        raise ValueError("data.dino_cache and data.cache must be separate directories")
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    dino_cfg = {**cfg, "person_encoder": {"backend": "dino"}}
    source = existing_cache(dino_cfg, root)
    if source is not None:
        source.validate_gallery(data.gallery_ids)
        print(f"Using existing DINO cache: {root}", flush=True)
    elif stage in ("persons", "clip"):
        raise ValueError("DINO cache missing/incomplete; run --cache-stage dino first")
    ready = None
    if backend == "fafa" and stage != "dino" and source is not None:
        ready = existing_cache(cfg, Path(data_cfg["cache"]), scene_root=root)
        if ready is not None:
            ready.validate_gallery(data.gallery_ids)
            print(f"Using existing FAFA cache: {data_cfg['cache']}", flush=True)
    if source is None:
        _prepare_dino(cfg, data, root)
    if backend == "fafa" and stage != "dino":
        if ready is None and stage != "clip":
            run_person_worker(cfg)
        if stage in ("all", "clip"):
            from rcr.proposed.cache.clip import build_clip_cache

            source = GalleryCache(data_cfg["cache"], scene_root=root)
            check_cache_config(source, cfg)
            if cfg.get("cache", {}).get("release_fafa_assets", False):
                release_fafa_assets(cfg)
            build_clip_cache(cfg, data, source)
