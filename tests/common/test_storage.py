"""Disk failures retain published files and clean interrupted writes."""

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rcr.common.io import save_results
from rcr.common.storage import (
    DISK_RESERVE_BYTES,
    atomic_copy,
    atomic_torch_save,
    estimate_torch_bytes,
    remove_cache_temporaries,
)


def test_low_disk_keeps_published_file_and_cleans_stale_write(tmp_path, monkeypatch):
    path = tmp_path / "text.pt"
    torch.save({"old": torch.ones(2)}, path)
    before = path.read_bytes()
    temporary = tmp_path / "text.pt.tmp"
    temporary.write_bytes(b"unfinished")
    monkeypatch.setattr(
        "rcr.common.storage.shutil.disk_usage",
        lambda path: SimpleNamespace(free=DISK_RESERVE_BYTES),
    )
    with pytest.raises(RuntimeError, match="notebook reserve"):
        atomic_torch_save({"new": torch.zeros(2)}, path)
    assert path.read_bytes() == before
    assert not temporary.exists()


@pytest.mark.parametrize("operation", ["save", "rename"])
def test_failed_atomic_save_keeps_old_file(tmp_path, monkeypatch, operation):
    path = tmp_path / "index.pt"
    torch.save({"old": torch.ones(2)}, path)
    before = path.read_bytes()
    if operation == "save":

        def fail(value, temporary):
            Path(temporary).write_bytes(b"partial write")
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(torch, "save", fail)
    else:

        def fail(source, destination):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError) as error:
        atomic_torch_save({"new": torch.zeros(2)}, path)
    assert error.value.errno == errno.ENOSPC
    assert path.read_bytes() == before
    assert not (tmp_path / "index.pt.tmp").exists()


def test_storage_budget_accounts_for_views_without_counting_aliases_twice(tmp_path):
    parent = torch.zeros(1024, 1024)
    value = {"slice": parent[0, :2], "alias": parent}
    # torch.save serializes the whole shared storage, even for a two-element view.
    path = tmp_path / "view.pt"
    torch.save(value, path)
    estimate = estimate_torch_bytes(value)
    assert path.stat().st_size <= estimate < 2 * parent.untyped_storage().nbytes()


def test_cleanup_removes_only_known_cache_temporaries(tmp_path):
    (tmp_path / "features").mkdir()
    stale = (".build.pt.tmp", "text.pt.tmp", "features/12.pt.tmp")
    retained = ("index.pt", ".build.pt", "features/12.pt", "features/notes.pt.tmp")
    for name in (*stale, *retained):
        (tmp_path / name).write_bytes(name.encode())
    remove_cache_temporaries(tmp_path)
    assert all(not (tmp_path / name).exists() for name in stale)
    assert all((tmp_path / name).read_bytes() == name.encode() for name in retained)


def test_failed_checkpoint_copy_keeps_best(tmp_path, monkeypatch):
    source, best = tmp_path / "last.pt", tmp_path / "best.pt"
    source.write_bytes(b"new checkpoint")
    best.write_bytes(b"previous best")

    def fail(source, temporary):
        Path(temporary).write_bytes(b"partial copy")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("rcr.common.storage.shutil.copyfile", fail)
    with pytest.raises(OSError):
        atomic_copy(source, best)
    assert source.read_bytes() == b"new checkpoint"
    assert best.read_bytes() == b"previous best"
    assert not (tmp_path / "best.pt.tmp").exists()


def test_low_disk_does_not_archive_current_retrieval_results(tmp_path, monkeypatch):
    current = {"rankings.pt": b"rankings", "run.json": b"{}", "metrics.json": b"{}"}
    for name, value in current.items():
        (tmp_path / name).write_bytes(value)
    monkeypatch.setattr(
        "rcr.common.storage.shutil.disk_usage", lambda p: SimpleNamespace(free=0)
    )
    with pytest.raises(RuntimeError, match="retrieval rankings"):
        save_results(tmp_path, {"rankings": torch.arange(10)}, {})
    assert all(
        (tmp_path / name).read_bytes() == value for name, value in current.items()
    )
    assert not (tmp_path / ".history").exists()
