"""Disk reservations shared by downloads and tensor writers."""

import errno
import shutil
from pathlib import Path


def require_disk_space(directory: Path, size: int) -> None:
    """Keep a margin for metadata, logs and filesystem overhead."""
    required = size + max(64 * 1024**2, size // 10)
    free = shutil.disk_usage(directory).free
    if free < required:
        raise OSError(
            errno.ENOSPC,
            f"Not enough disk space in {directory}: need about "
            f"{required / 1024**3:.2f} GiB free, found {free / 1024**3:.2f} GiB. "
            "Free space or move the cache/run before retrying.",
        )
