"""Download failures must not fill the disk or publish partial model artifacts."""

import errno
import io
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import gdown
import pytest

from rcr.baselines import prepare


@pytest.fixture
def fafa(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "rcr.baselines.fafa",
        SimpleNamespace(
            assets_ready=lambda _: True,
            official_source=lambda *a, **k: calls.append("source"),
            prepare_runtime_assets=lambda *a, **k: calls.append("runtime"),
        ),
    )
    cfg = {
        "checkpoint": {
            "path": str(tmp_path / "weights" / "released.pt"),
            "cache_root": str(tmp_path / "runtime"),
            "source_url": "https://drive.google.com/file/d/released-id/view",
        }
    }
    return cfg, calls


@pytest.mark.parametrize("local_path", [None, ""])
def test_fafa_download_owns_stream_and_removes_legacy_gdown_partials(
    fafa, monkeypatch, local_path
):
    cfg, calls = fafa
    cfg["checkpoint"]["local_path"] = local_path
    path = prepare.Path(cfg["checkpoint"]["path"])
    path.parent.mkdir()
    for name in ("released.part", "released.partAbc.tmp", "released.partXyz.part"):
        (path.parent / name).write_bytes(b"old partial")
    unrelated = path.parent / "other.part"
    unrelated.write_bytes(b"keep")
    # A crash after staging a local symlink must not let a later download
    # follow that link and overwrite its original weights.
    path.with_suffix(".pt.part").symlink_to(unrelated)

    def download(*, url, output, quiet):
        assert url == "https://drive.google.com/uc?id=released-id"
        assert not isinstance(output, str)  # gdown must not create another temp file.
        output.write(b"complete weights")
        return output

    monkeypatch.setattr(gdown, "download", download)
    prepare.prepare_fafa_model(cfg)
    assert path.read_bytes() == b"complete weights"
    assert unrelated.read_bytes() == b"keep"
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "other.part",
        "released.pt",
        "released.pt.lock",
    ]
    assert calls == ["source", "runtime"]


def test_real_gdown_downloads_into_owned_stream(fafa):
    cfg, _ = fafa
    payload = b"local test weights" * 100

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg["checkpoint"]["source_url"] = (
            f"http://127.0.0.1:{server.server_port}/weights"
        )
        prepare.prepare_fafa_model(cfg)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    path = prepare.Path(cfg["checkpoint"]["path"])
    assert path.read_bytes() == payload
    assert not list(path.parent.glob("*.part*"))


def test_failed_forced_download_preserves_existing_checkpoint(fafa, monkeypatch):
    cfg, calls = fafa
    cfg["checkpoint"]["runtime_reserve_gib"] = 0
    path = prepare.Path(cfg["checkpoint"]["path"])
    path.parent.mkdir()
    path.write_bytes(b"original")

    def fail(*, output, **_):
        output.write(b"partial replacement")
        raise OSError(errno.ENOSPC, "simulated disk full")

    monkeypatch.setattr(gdown, "download", fail)
    with pytest.raises(OSError, match="simulated disk full"):
        prepare.prepare_fafa_model(cfg, force=True)
    assert path.read_bytes() == b"original"
    assert not list(path.parent.glob("*.part*"))
    assert calls == ["source"]


def test_existing_and_mounted_checkpoints_skip_network_and_copy(fafa, monkeypatch):
    cfg, calls = fafa
    source = prepare.Path(cfg["checkpoint"]["path"]).parents[1] / "mounted.pt"
    source.write_bytes(b"mounted weights")
    cfg["checkpoint"]["local_path"] = str(source)
    monkeypatch.setattr(
        gdown, "download", lambda **_: pytest.fail("unexpected download")
    )
    prepare.prepare_fafa_model(cfg)
    path = prepare.Path(cfg["checkpoint"]["path"])
    assert path.is_symlink() and path.resolve() == source
    assert source.read_bytes() == b"mounted weights"
    prepare.prepare_fafa_model(cfg)
    assert calls == ["source", "runtime", "source", "runtime"]


def test_local_checkpoint_conflict_and_broken_link_fail_before_network(fafa):
    cfg, calls = fafa
    path = prepare.Path(cfg["checkpoint"]["path"])
    path.parent.mkdir()
    path.symlink_to(path.parent / "missing.pt")
    with pytest.raises(FileNotFoundError, match="Broken FAFA checkpoint"):
        prepare.prepare_fafa_model(cfg)
    path.unlink()
    path.write_bytes(b"existing")
    source = path.parent / "different.pt"
    source.write_bytes(b"different")
    cfg["checkpoint"]["local_path"] = str(source)
    with pytest.raises(ValueError, match="differs"):
        prepare.prepare_fafa_model(cfg)
    assert path.read_bytes() == b"existing"
    assert calls == []


def test_disk_preflight_stops_before_cloning_or_downloading(fafa, monkeypatch):
    cfg, calls = fafa
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    monkeypatch.setattr(
        gdown, "download", lambda **_: pytest.fail("unexpected download")
    )
    with pytest.raises(OSError) as error:
        prepare.prepare_fafa_model(cfg)
    assert error.value.errno == errno.ENOSPC
    assert calls == []
    assert not prepare.Path(cfg["checkpoint"]["path"]).exists()


def test_checkpoint_and_runtime_reservations_share_filesystem_budget(fafa, monkeypatch):
    cfg, _ = fafa
    reservations = []
    monkeypatch.setattr(
        prepare, "require_disk_space", lambda p, n: reservations.append(n)
    )
    prepare._check_fafa_space(cfg, download_checkpoint=True, runtime_ready=False)
    assert reservations == [9 * 1024**3]
    reservations.clear()
    prepare._check_fafa_space(cfg, download_checkpoint=False, runtime_ready=True)
    assert reservations == []


@pytest.mark.parametrize("value", [-1, True, "3", float("nan"), float("inf")])
def test_invalid_disk_estimate_rejected(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        prepare._reserve_gib({"reserve": value}, "reserve", 3)


def test_http_truncation_and_space_failure_never_replace_complete_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "clip.pt"
    path.write_bytes(b"previous")

    class Response(io.BytesIO):
        headers = {"Content-Length": "100"}

    monkeypatch.setattr(prepare, "urlopen", lambda _: Response(b"short response"))
    with pytest.raises(OSError, match="Incomplete download"):
        prepare.download("https://example.org/weights", path, force=True)
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.part"))
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Not enough disk space"):
        prepare.download("https://example.org/weights", path, force=True)
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.part"))
