"""Prepare external baseline artifacts explicitly; never install packages."""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

import yaml
from tqdm import tqdm

from rcr.methods.common.results import sha256_file


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
    import clip

    url = clip._MODELS[model_name]
    download(url, Path(path), force=force, expected_hash=url.split("/")[-2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace prepared weights and recheck runtime assets",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["method"] == "clip":
        prepare_clip(cfg["model"]["name"], cfg["model"]["checkpoint"], args.force)
    elif cfg["method"] == "fafa":
        import gdown

        from rcr.methods.baselines.fafa import official_source, prepare_runtime_assets

        official_source(cfg, prepare=True)
        checkpoint = Path(cfg["checkpoint"]["path"])
        if args.force or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".part")
            try:
                result = gdown.download(
                    url=cfg["checkpoint"]["source_url"],
                    output=str(temporary),
                    fuzzy=True,
                    quiet=False,
                )
                if (
                    result is None
                    or not temporary.is_file()
                    or not temporary.stat().st_size
                ):
                    raise RuntimeError(
                        "FAFA checkpoint download did not produce a file"
                    )
                os.replace(temporary, checkpoint)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        selector = cfg["localization"]["query_selector"]
        prepare_clip(selector["model"], selector["checkpoint"], args.force)
        detector = cfg["localization"]["detector"]
        download(
            detector["source_url"],
            Path(detector["checkpoint"]),
            force=args.force,
            expected_hash=detector["hash_prefix"],
        )
        prepare_runtime_assets(cfg, force=args.force)
    else:
        parser.error(f"unknown baseline {cfg['method']!r}")
    print("Baseline artifacts ready", flush=True)


if __name__ == "__main__":
    main()
