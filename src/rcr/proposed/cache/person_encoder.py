"""Prepare and load the frozen FAFA person encoder for feature extraction."""

from __future__ import annotations

import gc
import json
import math
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import torch

from rcr.common.torch_io import require_disk_space


def official_source(cfg: dict, *, prepare: bool = False) -> Path:
    source = cfg["source"]
    checkout = Path(source["local_checkout"]).resolve()
    if not checkout.exists():
        if not prepare:
            raise FileNotFoundError(
                "FAFA source missing; run the prepare command first"
            )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", source["repository"], str(checkout)], check=True
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    real_changes = [
        line for line in dirty.splitlines() if not line.endswith((".pyc", ".pyo"))
    ]
    if real_changes:
        raise RuntimeError(
            "Pinned FAFA source has tracked edits:\n" + "\n".join(real_changes)
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != source["commit"] and prepare:
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "origin", source["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", source["commit"]],
            check=True,
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    if actual != source["commit"]:
        raise RuntimeError("FAFA source commit mismatch; run the prepare command")
    directory = checkout / source.get("subdir", "FAFA_SynCPR")
    if not (directory / "src").is_dir():
        raise FileNotFoundError(directory / "src")
    return directory


def runtime_cache(cfg: dict, *, offline: bool) -> Path:
    root = Path(cfg["checkpoint"]["cache_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(root / "torch")
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(root / "xdg")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if offline:
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)
    return root


def official_api(directory: Path):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    src = str(directory / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from data_utils import squarepad_transform_test
    from lavis.models import load_model_and_preprocess

    return squarepad_transform_test, load_model_and_preprocess


def marker_identity(cfg: dict) -> dict:
    return {
        "source_commit": cfg["source"]["commit"],
        "model_name": cfg["checkpoint"]["model_name"],
        "model_type": cfg["checkpoint"]["model_type"],
    }


def assets_ready(cfg: dict) -> bool:
    marker = Path(cfg["checkpoint"]["runtime_assets_marker"])
    root = Path(cfg["checkpoint"]["cache_root"])
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in marker_identity(cfg).items()):
            return False
        files = saved["files"]
        return bool(files) and all(
            (root / item["path"]).is_file()
            and (root / item["path"]).stat().st_size == item["size"]
            for item in files
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def prepare_runtime_assets(cfg: dict, *, force: bool = False) -> None:
    directory = official_source(cfg, prepare=True)
    if assets_ready(cfg) and not force:
        print("FAFA runtime assets already prepared", flush=True)
        return
    root = runtime_cache(cfg, offline=False)
    _, load_model = official_api(directory)
    print("Preparing official FAFA runtime assets on CPU", flush=True)
    model, _, _ = load_model(
        name=cfg["checkpoint"]["model_name"],
        model_type=cfg["checkpoint"]["model_type"],
        is_eval=True,
        device="cpu",
    )
    del model
    gc.collect()
    files = [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise RuntimeError("Official model loaded but the runtime cache is empty")
    marker = Path(cfg["checkpoint"]["runtime_assets_marker"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({**marker_identity(cfg), "files": files}, indent=2), encoding="utf-8"
    )


@contextmanager
def offline_model_load():
    """Fail on missing runtime assets instead of downloading during inference."""
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def blocked(self, address):
        raise RuntimeError("Network blocked in FAFA inference; run the prepare command")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def text_policy(cfg: dict) -> dict[str, int]:
    """Resolve query-only text limits from the pinned FAFA defaults.

    The loader defaults to 32 tokens and its caption processor to 50 words. A
    word cap at least as large as the token cap leaves truncation to the tokenizer.
    """
    length = cfg.get("checkpoint", {}).get("max_txt_len", 32)
    if type(length) is not int or not 2 <= length <= 512:
        raise ValueError("checkpoint.max_txt_len must be an integer in [2, 512]")
    return {"max_txt_len": length, "max_words": max(50, length)}


def load_fafa(cfg: dict, device):
    policy = text_policy(cfg)
    directory = official_source(cfg)
    if not assets_ready(cfg):
        raise RuntimeError("FAFA runtime assets missing/stale; run the prepare command")
    runtime_cache(cfg, offline=True)
    squarepad, load_model = official_api(directory)
    with offline_model_load():
        model, _, processors = load_model(
            name=cfg["checkpoint"]["model_name"],
            model_type=cfg["checkpoint"]["model_type"],
            is_eval=True,
            device=str(device),
        )
    # The authors' released checkpoint is a trusted prepared model artifact.
    checkpoint = torch.load(
        cfg["checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not set(state) & set(model.state_dict()):
        raise ValueError("FAFA checkpoint has no matching model parameters")
    incompatible = model.load_state_dict(
        state, strict=False
    )  # Official inference policy.
    model.fda_k = int(cfg["checkpoint"]["fda_k"])
    model.fda_alpha = float(cfg["checkpoint"]["fda_alpha"])
    model.use_soft = bool(cfg["checkpoint"]["use_soft"])
    model.max_txt_len = policy["max_txt_len"]
    processors["eval"].max_words = policy["max_words"]
    print(
        f"FAFA text: max_txt_len={model.max_txt_len} tokens, "
        f"processor.max_words={processors['eval'].max_words}",
        flush=True,
    )
    preprocess = squarepad(
        int(cfg["checkpoint"]["image_size"]),
        need_size=tuple(cfg["checkpoint"]["test_resize_hw"]),
    )
    return (
        model.to(device).eval(),
        processors,
        preprocess,
        {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "text_policy": policy,
        },
    )


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
    from filelock import FileLock

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
            # Publish the checkpoint atomically from a single download stream.
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
