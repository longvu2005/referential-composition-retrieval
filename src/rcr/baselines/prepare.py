"""Prepare external baseline artifacts explicitly; never install packages."""

import hashlib
import math
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from filelock import FileLock
from tqdm import tqdm

from rcr.common.io import sha256_file
from rcr.common.torch_io import require_disk_space


def require_shared_detection_cache(cfg: dict) -> None:
    """Fail before FAFA setup when its immutable detector cache is absent."""
    root = Path(cfg["localization"]["detector"]["cache_dir"])
    if not (root / "index.pt").is_file() or (root / ".building").exists():
        raise FileNotFoundError(
            f"FAFA requires the completed shared person-box cache at {root}; run "
            "`.venv-proposed/bin/python tools/run.py build-cache "
            "--config configs/proposed.yaml --cache-stage detections` first"
        )
    from rcr.common.detection import DetectionCache

    cache = DetectionCache(root)
    detector = cfg["localization"]["detector"]
    if detector["backend"] != "grounding_dino_cache":
        raise ValueError("use the shared grounding_dino_cache backend")
    spec = {key: detector[key] for key in ("model", "threshold", "text_threshold")}
    if "revision" in detector:
        spec["revision"] = detector["revision"]
    cache.check_config(spec)


def download(url: str, path: Path, *, force: bool, expected_hash: str | None = None):
    if (
        path.is_file()
        and path.stat().st_size
        and not force
        and (expected_hash is None or sha256_file(path).startswith(expected_hash))
    ):
        print(f"Already prepared: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "rcr-baselines/0.1"})
        ) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            require_disk_space(path.parent, total or 0)
            received = 0
            with (
                temporary.open("wb") as handle,
                tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar,
            ):
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    handle.write(chunk)
                    received += len(chunk)
                    bar.update(len(chunk))
            if total is not None and received != total:
                raise OSError(
                    f"Incomplete download for {path}: {received}/{total} bytes"
                )
        if expected_hash and not digest.hexdigest().startswith(expected_hash):
            raise ValueError(f"Checksum mismatch for {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_clip(model_name: str, path: str, force: bool):
    from clip.clip import _MODELS

    url = _MODELS[model_name]
    download(url, Path(path), force=force, expected_hash=url.split("/")[-2])


def _reserve_gib(settings: dict, key: str, default: float) -> int:
    value = settings.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite nonnegative GiB")
    return int(value * 1024**3)


def _check_fafa_space(cfg: dict, *, download_checkpoint: bool, runtime_ready: bool):
    """Reserve checkpoint/runtime together when they share one filesystem."""
    settings = cfg["checkpoint"]
    requests = []
    if download_checkpoint:
        requests.append(
            (
                Path(settings["path"]).parent,
                _reserve_gib(settings, "download_reserve_gib", 3),
            )
        )
    if not runtime_ready:
        requests.append(
            (
                Path(settings["cache_root"]),
                _reserve_gib(settings, "runtime_reserve_gib", 6),
            )
        )
    volumes = {}
    for directory, size in requests:
        if not size:
            continue
        directory = directory.absolute()
        while not directory.exists():
            directory = directory.parent
        device = directory.stat().st_dev
        previous = volumes.get(device, (directory, 0))
        volumes[device] = previous[0], previous[1] + size
    for directory, size in volumes.values():
        require_disk_space(directory, size)


def prepare_fafa_model(cfg: dict, *, force: bool = False) -> None:
    """Prepare FAFA from one local mount or a streamed, atomic download."""
    import gdown

    from rcr.baselines.fafa import assets_ready, official_source, prepare_runtime_assets

    settings = cfg["checkpoint"]
    checkpoint = Path(settings["path"])
    if checkpoint.is_symlink() and not checkpoint.is_file():
        raise FileNotFoundError(f"Broken FAFA checkpoint link: {checkpoint}")
    local = (
        Path(settings["local_path"]).resolve() if settings.get("local_path") else None
    )
    if local is not None:
        if not local.is_file() or not local.stat().st_size:
            raise FileNotFoundError(f"Missing or empty local FAFA checkpoint: {local}")
        if checkpoint.exists() and checkpoint.resolve() != local:
            raise ValueError(
                "existing FAFA checkpoint differs from checkpoint.local_path"
            )
    need_download = local is None and (
        force or not checkpoint.is_file() or not checkpoint.stat().st_size
    )
    _check_fafa_space(
        cfg,
        download_checkpoint=need_download,
        runtime_ready=assets_ready(cfg) and not force,
    )
    official_source(cfg, prepare=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(checkpoint) + ".lock"):
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".part")
        if local is not None and not checkpoint.exists():
            temporary.unlink(missing_ok=True)
            try:
                temporary.symlink_to(local)
                os.replace(temporary, checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
        elif need_download and (
            force or not checkpoint.is_file() or not checkpoint.stat().st_size
        ):
            # Own the stream, rather than letting gdown create another large
            # partial file. Remove only this checkpoint's interrupted partials.
            for partial in checkpoint.parent.glob(
                checkpoint.with_suffix(".part").name + "*"
            ):
                partial.unlink()
            temporary.unlink(missing_ok=True)
            try:
                source_url = settings["source_url"]
                parsed = urlparse(source_url)
                if parsed.hostname == "drive.google.com" and parsed.path.startswith(
                    "/file/d/"
                ):
                    source_url = (
                        f"https://drive.google.com/uc?id={parsed.path.split('/')[3]}"
                    )
                with temporary.open("wb") as handle:
                    result = gdown.download(url=source_url, output=handle, quiet=False)
                if result is None or not temporary.stat().st_size:
                    raise RuntimeError(
                        "FAFA checkpoint download did not produce a file"
                    )
                os.replace(temporary, checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
    prepare_runtime_assets(cfg, force=force)


def prepare(cfg: dict, *, force: bool = False) -> None:
    if cfg["method"] == "clip":
        prepare_clip(cfg["model"]["name"], cfg["model"]["checkpoint"], force)
    elif cfg["method"] == "fafa":
        require_shared_detection_cache(cfg)
        prepare_fafa_model(cfg, force=force)
        selector = cfg["localization"]["query_selector"]
        prepare_clip(selector["model"], selector["checkpoint"], force)
    else:
        raise ValueError(f"unknown baseline {cfg['method']!r}")
    print("Baseline artifacts ready", flush=True)
