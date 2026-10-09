"""Budget large writes and retain notebook space on shared filesystems."""

import shutil
from pathlib import Path

DISK_RESERVE_BYTES = 1024**3


def format_bytes(value: int) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def require_free_space(path, additional_bytes: int, *, context: str) -> None:
    """Check additional peak bytes, including space reserved for notebook saves."""
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = additional_bytes + DISK_RESERVE_BYTES
    if free < required:
        raise RuntimeError(
            f"Not enough disk space for {context} at {probe}: "
            f"free={format_bytes(free)}, estimated additional write="
            f"{format_bytes(additional_bytes)}, notebook reserve="
            f"{format_bytes(DISK_RESERVE_BYTES)}. "
            "Free space or use a larger writable volume, keep completed cache "
            "shards, then resume build-cache."
        )


def estimate_torch_bytes(value) -> int:
    """Estimate serialization without copying tensors; account for shared storage."""
    import torch

    storages = {}
    visited = set()

    def size(item):
        if id(item) in visited:
            return 0
        visited.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(item.device, storage.data_ptr())] = storage.nbytes()
            return 512
        if isinstance(item, dict):
            return 64 + sum(size(k) + size(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return 64 + sum(size(v) for v in item)
        if isinstance(item, str):
            return 64 + len(item.encode("utf-8"))
        if isinstance(item, bytes):
            return 64 + len(item)
        return 64

    metadata = size(value)
    return sum(storages.values()) + metadata + 64 * 1024


def atomic_torch_save(value, path) -> None:
    """Keep the published file on failure and remove only our temporary write."""
    import torch

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    require_free_space(
        path.parent, estimate_torch_bytes(value), context=f"writing {path.name}"
    )
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_cache_temporaries(root) -> None:
    """Remove interrupted cache writes after the build signature is validated."""
    root = Path(root)
    paths = [
        root / name
        for name in (
            ".build.pt.tmp", "index.pt.tmp", "text.pt.tmp", "supervision.pt.tmp"
        )
    ]
    paths.extend(
        path
        for path in (root / "features").glob("*.pt.tmp")
        if path.name.removesuffix(".pt.tmp").isdecimal()
    )
    for path in paths:
        if path.is_file():
            path.unlink()


def atomic_copy(source, destination) -> None:
    """Publish checkpoint copies without truncating an existing best checkpoint."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    require_free_space(
        destination.parent, source.stat().st_size, context=f"copying {destination.name}"
    )
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
