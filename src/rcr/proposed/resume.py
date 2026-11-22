"""Durable training progress, compatibility checks and safe interruption points."""

import fcntl
import json
import random
import signal
import sys
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from rcr.common.data import split_fingerprint
from rcr.common.io import atomic_write_text, sha256_file
from rcr.evaluation.provenance import digest_json
from rcr.proposed.contracts import ARCHITECTURE_VERSION, CALIBRATED_RETRIEVAL_KEYS

RESUME_VERSION = 1


def require_new_run(output: Path) -> None:
    if any((output / name).exists() for name in ("config.yaml", "best.pt", "last.pt")):
        raise FileExistsError(
            f"training run already exists at {output}; use --resume or a new output.dir"
        )


def checkpoint_version(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass
class TrainingProgress:
    epoch: int = 0
    phase: str = "epoch_end"
    next_batch: int = 0
    global_step: int = 0
    state_pairs_seen: int = 0
    state_pairs_epoch: int = 0
    best: dict = field(
        default_factory=lambda: {
            "epoch": None,
            "full_map": None,
            "macro_full_map": None,
            "score": None,
            **dict.fromkeys(CALIBRATED_RETRIEVAL_KEYS),
        }
    )
    hard_pools: dict | None = None
    history: list = field(default_factory=list)
    batches: list = field(default_factory=list)
    sampling_rng: torch.Tensor | None = None
    identity_rng: torch.Tensor | None = None
    totals: dict = field(default_factory=dict)
    crop_totals: dict = field(default_factory=dict)
    candidate_totals: dict = field(default_factory=dict)
    wandb_id: str | None = None
    wandb_project: str | None = None


def capture_rng(*, cuda: bool = False) -> dict:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def resume_signature(cfg, data, cache, device, amp) -> dict:
    """Bind numerical settings, training inputs, implementation and runtime."""
    config = deepcopy(cfg)
    for key in ("checkpoint", "output", "wandb", "checkpointing", "clip_text_stats"):
        config.pop(key, None)
    for key in (
        "final_dir",
        "image_root",
        "detection_cache",
        "scene_cache",
        "fafa_image_cache",
        "fafa_query_cache",
    ):
        config["data"].pop(key, None)
    config.get("cache", {}).pop("supervision_cache", None)
    package = Path(__file__).resolve().parents[1]
    files = {
        str(path.relative_to(package)): sha256_file(path)
        for path in sorted(package.rglob("*.py"))
        if not path.name.startswith(".")
    }
    packages = {}
    for name in ("torch", "numpy", "scipy", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "config": config,
        "source_sha256": digest_json(files),
        "inputs": {
            "train": split_fingerprint(data, "train"),
            "val": split_fingerprint(data, "val"),
            "gallery": digest_json(data.gallery_ids),
            "cache_id": cache.cache_id,
            "scene_cache_id": cache.scene_cache_id,
            "detection_cache_id": cache.detection_cache_id,
            "native_cache": cache.native.metadata,
            "encoder_metadata": cache.encoder_metadata,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "packages": packages,
            "device": str(device),
            "amp": amp,
            "cuda": torch.version.cuda if device.type == "cuda" else None,
            "cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
    }


def load_resume(path: Path, output: Path) -> dict:
    """Reject inference-only checkpoints before changing any existing run files."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("training_state")
    required = {"version", "progress", "optimizer", "scaler", "rng", "signature"}
    if not isinstance(state, dict) or required - state.keys():
        raise ValueError(
            "checkpoint has no complete optimizer/scaler/RNG training state; "
            "inference-only and legacy checkpoints cannot resume exactly"
        )
    if state["version"] != RESUME_VERSION:
        raise ValueError("unsupported training checkpoint version")
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "resume requires the same architecture; train in a new output.dir"
        )
    if (
        not isinstance(state["optimizer"], dict)
        or {"state", "param_groups"} - state["optimizer"].keys()
        or not isinstance(state["scaler"], dict)
        or not isinstance(state["rng"], dict)
        or {"python", "numpy", "torch", "cuda"} - state["rng"].keys()
        or not isinstance(state["signature"], dict)
        or {"config", "source_sha256", "inputs", "runtime"} - state["signature"].keys()
    ):
        raise ValueError("checkpoint has incomplete optimizer/scaler/RNG state")
    if path.resolve() != (output / "last.pt").resolve():
        raise ValueError(
            "resume from last.pt in output.dir; keep the complete run directory"
        )
    progress = TrainingProgress(**state["progress"])
    if (
        not 0 <= progress.epoch <= checkpoint["config"]["train"]["epochs"]
        or progress.global_step < 0
        or progress.phase not in ("train", "validation", "epoch_end")
        or not 0 <= progress.next_batch <= len(progress.batches)
        or (
            progress.phase != "epoch_end"
            and (
                progress.epoch < 1
                or not progress.batches
                or not isinstance(progress.sampling_rng, torch.Tensor)
                or not isinstance(progress.identity_rng, torch.Tensor)
            )
        )
        or (
            progress.phase == "validation"
            and progress.next_batch != len(progress.batches)
        )
        or [row["epoch"] for row in progress.history]
        != list(range(1, progress.epoch + (progress.phase == "epoch_end")))
    ):
        raise ValueError("checkpoint has inconsistent training progress")
    return checkpoint


def verify_resume(checkpoint: dict, signature: dict) -> TrainingProgress:
    saved = checkpoint["training_state"]["signature"]
    changed = [key for key in signature if saved.get(key) != signature[key]]
    if changed:
        raise ValueError("resume checkpoint differs in " + ", ".join(changed))
    return TrainingProgress(**checkpoint["training_state"]["progress"])


def verify_best(output: Path, progress: TrainingProgress) -> None:
    """Check committed selections; a pending validation may be republished on resume."""
    if progress.best["epoch"] is None or progress.phase == "validation":
        return
    selected = output / "selected.yaml"
    best_path = output / "best.pt"
    if not selected.is_file() or not best_path.is_file():
        raise ValueError("resume requires the run's best.pt and selected.yaml")
    from rcr.common.config import load_config

    config = load_config(selected)
    if Path(config["checkpoint"]).resolve() != best_path.resolve():
        raise ValueError(
            "selected.yaml must point to best.pt in the same run directory"
        )
    if config.get("selected_checkpoint_sha256") != sha256_file(best_path):
        raise ValueError("best.pt changed since its validation selection")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["epoch"] != progress.best["epoch"]
        or checkpoint["best_score"] != progress.best["score"]
        or any(
            config["retrieval"][key] != progress.best[key]
            for key in CALIBRATED_RETRIEVAL_KEYS
        )
    ):
        raise ValueError("best checkpoint disagrees with resumed validation progress")


@contextmanager
def training_lock(output: Path):
    """Release the process lock automatically, including after an abrupt exit."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another training process is using output.dir"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def training_state(progress, optimizer, scaler, signature) -> dict:
    return {
        "version": RESUME_VERSION,
        "progress": asdict(progress),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(cuda=bool(signature["runtime"]["cuda_devices"])),
        "signature": signature,
    }


def restore_history(output: Path, progress: TrainingProgress) -> None:
    """Keep epoch history aligned with the last atomically published checkpoint."""
    atomic_write_text(
        output / "history.jsonl",
        "".join(json.dumps(row) + "\n" for row in progress.history),
    )


def cleanup_validation_caches(output: Path, progress: TrainingProgress) -> None:
    """Drop transient rankings only for epochs committed in the training checkpoint."""
    for row in progress.history:
        path = (
            output
            / "evaluation"
            / f"epoch_{row['epoch']:03d}"
            / "pending_validation.pt"
        )
        path.unlink(missing_ok=True)


class TrainingControl:
    """Defer interrupt signals until an optimizer batch reaches a safe boundary."""

    def __init__(self, run):
        self.run = run
        self.stopping = False
        self.handlers = {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in (signal.SIGINT, signal.SIGTERM):
                self.handlers[name] = signal.getsignal(name)
                signal.signal(name, self._stop)
        return self

    def _stop(self, _signal, _frame):
        if self.stopping:
            raise KeyboardInterrupt
        self.stopping = True

    def __exit__(self, *_error):
        for name, handler in self.handlers.items():
            signal.signal(name, handler)
        if self.run is not None:
            self.run.finish()
