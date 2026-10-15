"""Atomic tensor files with a disk-space check before allocating a temporary."""

import errno
import os
import shutil
from pathlib import Path

import torch


def tensor_bytes(value) -> int:
    """Count distinct tensor storages, including shared views, once."""
    storages = {}

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(item.device, storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sum(storages.values())


def require_disk_space(directory: Path, size: int) -> None:
    """Keep a margin for serialization metadata, logs and filesystem overhead."""
    required = size + max(64 * 1024**2, size // 10)
    free = shutil.disk_usage(directory).free
    if free < required:
        raise OSError(
            errno.ENOSPC,
            f"Not enough disk space in {directory}: need about "
            f"{required / 1024**3:.2f} GiB free, found {free / 1024**3:.2f} GiB. "
            "Free space or move the cache/run before retrying.",
        )


def save_torch(value, path: Path) -> None:
    """Never replace a complete file with a partially written checkpoint/shard."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        require_disk_space(path.parent, tensor_bytes(value))
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def link_checkpoint(source: Path, target: Path) -> None:
    """A best/last hard link saves a copy; replacing last preserves an older best."""
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            require_disk_space(target.parent, source.stat().st_size)
            shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
