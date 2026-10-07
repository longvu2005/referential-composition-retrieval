"""Prepare external baseline artifacts explicitly; never install packages."""

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm

from rcr.common.io import sha256_file


def download(url: str, path: Path, *, force: bool, expected_hash: str | None = None):
    if path.is_file() and path.stat().st_size and not force:
        if expected_hash is None or sha256_file(path).startswith(expected_hash):
            print(f"Already prepared: {path}", flush=True)
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "rcr-baselines/0.1"})
        ) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with (
                temporary.open("wb") as handle,
                tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar,
            ):
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    handle.write(chunk)
                    bar.update(len(chunk))
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


def prepare_fafa_model(cfg: dict, *, force: bool = False) -> None:
    """Only FAFA weights/source/runtime; no CLIP selector or baseline detector."""
    import gdown

    from rcr.baselines.fafa import official_source, prepare_runtime_assets

    official_source(cfg, prepare=True)
    checkpoint = Path(cfg["checkpoint"]["path"])
    if force or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".part")
        try:
            source_url = cfg["checkpoint"]["source_url"]
            parsed = urlparse(source_url)
            if parsed.hostname == "drive.google.com" and parsed.path.startswith(
                "/file/d/"
            ):
                file_id = parsed.path.split("/")[3]
                source_url = f"https://drive.google.com/uc?id={file_id}"
            result = gdown.download(
                url=source_url,
                output=str(temporary),
                quiet=False,
            )
            if (
                result is None
                or not temporary.is_file()
                or not temporary.stat().st_size
            ):
                raise RuntimeError("FAFA checkpoint download did not produce a file")
            os.replace(temporary, checkpoint)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    prepare_runtime_assets(cfg, force=force)


def prepare(cfg: dict, *, force: bool = False) -> None:
    if cfg["method"] == "clip":
        prepare_clip(cfg["model"]["name"], cfg["model"]["checkpoint"], force)
    elif cfg["method"] == "fafa":
        prepare_fafa_model(cfg, force=force)
        selector = cfg["localization"]["query_selector"]
        prepare_clip(selector["model"], selector["checkpoint"], force)
        detector = cfg["localization"]["detector"]
        download(
            detector["source_url"],
            Path(detector["checkpoint"]),
            force=force,
            expected_hash=detector["hash_prefix"],
        )
    else:
        raise ValueError(f"unknown baseline {cfg['method']!r}")
    print("Baseline artifacts ready", flush=True)
