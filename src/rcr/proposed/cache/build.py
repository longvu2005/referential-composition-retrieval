"""Select cache stages and run the isolated FAFA worker."""

import subprocess
import tempfile
from pathlib import Path

import yaml

from rcr.baselines.fafa import text_policy
from rcr.common.config import load_config
from rcr.common.data import load_rcr_inputs
from rcr.common.detection import DetectionCache, build_detection_cache
from rcr.common.io import sha256_file
from rcr.common.runtime import resolve_device
from rcr.proposed.cache.native import NATIVE_VERSION, NativeQueryCache
from rcr.proposed.cache.scene import SceneCache, _prepare_scene
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.nn.encoders import check_clip_config


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
    if person_backend == "clip":
        if cache.encoder_metadata["image_encoder"] != cfg["image_encoder"]:
            raise ValueError("cache image encoder/preprocessing differs from config")
        return
    if cache._scene_cache is not None:
        check_cache_config(cache._scene_cache, cfg, person_backend="clip")
    if cache.format_version != 7 or cache.detections is None:
        raise ValueError("build a FAFA image cache bound to data.detection_cache")
    cache.detections.check_config(cfg["detector"])
    actual = cache.encoder_metadata["person_encoder"] or {}
    if actual.get("backend") != person_backend:
        raise ValueError("cache person encoder differs; use a matching cache directory")
    if cache.encoder_metadata["detector"] != cfg["detector"]:
        raise ValueError("cache detector differs from config")
    if person_backend == "fafa":
        if cache.native is None:
            raise ValueError("build a FAFA image cache in data.fafa_image_cache")
        native_path = cfg["person_encoder"].get("fafa_config")
        if native_path is None or not Path(native_path).is_file():
            return  # Cache-only train/retrieve uses the recorded native provenance.
        native_cfg = load_fafa_config(cfg)
        if actual.get("spec") != fafa_spec(native_cfg):
            raise ValueError("cache FAFA checkpoint/source/preprocessing spec differs")
        checkpoint = Path(native_cfg["checkpoint"]["path"])
        if checkpoint.is_file() and actual.get("checkpoint_sha256") != sha256_file(
            checkpoint
        ):
            raise ValueError("cache FAFA checkpoint weights differ")


def existing_cache(
    cfg: dict,
    root: Path,
    *,
    scene_root: Path | None = None,
    detection_root: Path | None = None,
    person_backend: str = "fafa",
):
    if not (root / "index.pt").is_file() or (root / ".building").exists():
        return None
    cache = (
        SceneCache(root)
        if person_backend == "clip"
        else GalleryCache(root, scene_root=scene_root, detection_root=detection_root)
    )
    check_cache_config(cache, cfg, person_backend=person_backend)
    return cache


def check_query_text_config(cache: GalleryCache, cfg: dict) -> None:
    """Check a present extraction config without requiring the FAFA runtime.

    Cache-only machines may omit the native YAML entirely. In that case its
    recorded text policy is authoritative and remains bound to the checkpoint.
    """
    if cache.native.metadata.get("version") != NATIVE_VERSION:
        return
    path = cfg.get("person_encoder", {}).get("fafa_config")
    if path is None or not Path(path).is_file():
        return
    if cache.native.text_policy != text_policy(load_config(path)):
        raise ValueError(
            "FAFA query text policy differs from config; "
            "run build-cache --cache-stage fafa-query before training/retrieval"
        )


def query_cache_ready(root: Path, image: GalleryCache, samples, policy: dict) -> bool:
    if not (root / "index.pt").is_file() or (root / ".building").exists():
        return False
    queries = NativeQueryCache(root, image.feature_signature, image.native.dim)
    return queries.text_policy == policy and queries.contains_samples(samples)


def cache_paths(cfg: dict) -> tuple[Path, Path, Path, Path]:
    """Keep every cache separate, including through symlinks and parent paths."""
    data = cfg["data"]
    if "dino_cache" in data:
        raise ValueError(
            "replace data.dino_cache with data.detection_cache and data.scene_cache"
        )
    if "cache" in data or "native_cache" in data:
        raise ValueError(
            "replace data.cache/data.native_cache with data.fafa_image_cache and "
            "data.fafa_query_cache, and add data.detection_cache; "
            "rebuild FAFA features for the separated crop contract"
        )
    keys = ("detection_cache", "scene_cache", "fafa_image_cache", "fafa_query_cache")
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"separated caches require data keys: {', '.join(missing)}")
    paths = tuple(Path(data[key]) for key in keys)
    resolved = [path.resolve() for path in paths]
    for i, first in enumerate(resolved):
        for second in resolved[i + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError(
                    "Detection, scene, FAFA image and query caches "
                    "must be separate directories"
                )
    return paths


def prepare_cache(cfg: dict, *, stage: str = "all") -> None:
    """Build each cache independently; never write to a completed image cache."""
    if stage not in ("all", "detections", "scene", "fafa", "fafa-image", "fafa-query"):
        raise ValueError("invalid cache stage")
    if stage in ("all", "scene"):
        check_clip_config(cfg)
    detection_root, root, image_root, query_root = cache_paths(cfg)
    data_cfg = cfg["data"]
    data = load_rcr_inputs(data_cfg["final_dir"], data_cfg["image_root"])
    if stage in ("all", "detections"):
        build_detection_cache(
            data.gallery_ids,
            [data.image_path(i) for i in data.gallery_ids],
            detection_root,
            cfg["detector"],
            device=resolve_device(cfg),
        )
    if stage == "detections":
        return
    if stage in ("all", "scene"):
        _prepare_scene(cfg, data, root)
    if stage == "scene":
        return
    detections = DetectionCache(detection_root)
    detections.validate_gallery(data.gallery_ids)
    detections.check_config(cfg["detector"])
    image = existing_cache(cfg, image_root, detection_root=detection_root)
    if image is not None:
        image.validate_gallery(data.gallery_ids)
    if stage == "fafa-image" and image is not None:
        print(f"Using existing FAFA image cache: {image_root}", flush=True)
        return
    if (
        stage != "fafa-image"
        and image is not None
        and query_cache_ready(
            query_root, image, data.samples, text_policy(load_fafa_config(cfg))
        )
    ):
        print(f"Using existing FAFA caches: {image_root}, {query_root}", flush=True)
        return
    run_person_worker(cfg, stage="fafa" if stage == "all" else stage)
    if stage != "fafa-query":
        image = existing_cache(cfg, image_root, detection_root=detection_root)
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
            queries = NativeQueryCache(
                query_root, image.feature_signature, image.native.dim
            )
            if queries.text_policy != text_policy(load_fafa_config(cfg)):
                raise RuntimeError("FAFA worker produced a different query text policy")
            queries.validate_samples(data.samples)
