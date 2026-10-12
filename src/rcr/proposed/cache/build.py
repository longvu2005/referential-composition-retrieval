"""Select cache stages and run the isolated FAFA worker."""

import subprocess
import tempfile
from pathlib import Path

import yaml

from rcr.common.config import load_config
from rcr.common.data import load_rcr_data
from rcr.common.io import sha256_file
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
    """Share files only; native FAFA stays in its own dependency environment."""
    if cfg.get("person_encoder", {}).get("backend", "dino") == "dino":
        return
    worker = Path(__file__).resolve().parents[4] / "tools/cache_fafa.py"
    with tempfile.TemporaryDirectory(prefix="rcr-person-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [cfg["person_encoder"]["python"], str(worker), "--config", str(path)]
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


def prepare_cache(cfg: dict, *, stage: str = "all") -> None:
    """Build DINO and/or FAFA independently; completed caches stay immutable."""
    backend = cfg.get("person_encoder", {}).get("backend", "dino")
    if backend not in ("dino", "fafa"):
        raise ValueError("person_encoder.backend must be dino or fafa")
    if stage not in ("all", "dino", "persons"):
        raise ValueError("cache stage must be all, dino or persons")
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
    elif stage == "persons":
        raise ValueError("DINO cache missing/incomplete; run --cache-stage dino first")
    if backend == "fafa" and stage != "dino" and source is not None:
        ready = existing_cache(cfg, Path(data_cfg["cache"]), scene_root=root)
        if ready is not None:
            ready.validate_gallery(data.gallery_ids)
            print(f"Using existing FAFA cache: {data_cfg['cache']}", flush=True)
            return
    if source is None:
        _prepare_dino(cfg, data, root)
    if backend == "fafa" and stage != "dino":
        run_person_worker(cfg)
