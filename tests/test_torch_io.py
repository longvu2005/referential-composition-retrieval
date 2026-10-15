"""A failed save must preserve checkpoints and remove partial temporary files."""

import errno
from types import SimpleNamespace

import pytest
import torch

from rcr.common import torch_io


def test_tensor_storage_is_counted_once_for_shared_views():
    value = torch.arange(12.0)
    assert torch_io.tensor_bytes({"a": value, "b": [value[:2], value]}) == 48


def test_full_disk_is_reported_before_touching_a_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous checkpoint")
    monkeypatch.setattr(
        torch_io.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    monkeypatch.setattr(torch, "save", lambda *a: pytest.fail("attempted disk write"))
    with pytest.raises(OSError, match="Not enough disk space") as error:
        torch_io.save_torch({"x": torch.ones(3)}, path)
    assert error.value.errno == errno.ENOSPC
    assert path.read_bytes() == b"previous checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()


def test_failed_serialization_removes_temporary_and_preserves_old_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous checkpoint")

    def fail(value, temporary):
        temporary.write_bytes(b"partial write")
        raise RuntimeError("iostream error")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="iostream error"):
        torch_io.save_torch({"x": torch.ones(3)}, path)
    assert path.read_bytes() == b"previous checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()


def test_best_link_keeps_its_epoch_when_last_is_replaced(tmp_path):
    last, best = tmp_path / "last.pt", tmp_path / "best.pt"
    torch_io.save_torch({"epoch": 1}, last)
    torch_io.link_checkpoint(last, best)
    assert last.samefile(best)
    torch_io.save_torch({"epoch": 2}, last)
    assert torch.load(best, weights_only=True)["epoch"] == 1
    assert torch.load(last, weights_only=True)["epoch"] == 2
    assert not last.samefile(best)
    torch_io.link_checkpoint(last, best)
    assert last.samefile(best)


def test_best_copy_fallback_is_atomic_when_links_are_unavailable(tmp_path, monkeypatch):
    last, best = tmp_path / "last.pt", tmp_path / "best.pt"
    torch_io.save_torch({"epoch": 4}, last)

    def unsupported(*args):
        raise OSError(errno.EPERM, "hard links unavailable")

    monkeypatch.setattr(torch_io.os, "link", unsupported)
    torch_io.link_checkpoint(last, best)
    assert torch.load(best, weights_only=True)["epoch"] == 4
    assert not best.with_suffix(".pt.tmp").exists()
