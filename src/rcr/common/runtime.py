"""Small runtime helpers shared by training and retrieval."""

import hashlib
import os
import random
from contextlib import contextmanager

import numpy as np
import torch

# cuBLAS needs a fixed workspace size to be reproducible on CUDA >= 10.2.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def resolve_device(cfg):
    name = cfg["runtime"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and cfg["runtime"].get("deterministic", True):
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
        if workspace not in (CUBLAS_WORKSPACE_CONFIG, ":16:8"):
            raise ValueError(
                "deterministic CUDA runs require CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8; set it before starting Python"
            )
    return device


def derive_seed(seed: int, stream: str, index: int = 0) -> int:
    """Map ``(run seed, named stream, index)`` to an independent 63-bit seed.

    Adding an epoch to the run seed makes neighbouring seeds replay each other's
    streams (seed 0 epoch 2 equals seed 1 epoch 1). Hashing keeps every triple
    distinct and, unlike ``hash()``, is identical in every process.
    """
    digest = hashlib.sha256(f"{seed}:{stream}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def derived_generator(seed: int, stream: str, index: int = 0) -> torch.Generator:
    """Return a fresh CPU generator for one named stream of a run."""
    return torch.Generator().manual_seed(derive_seed(seed, stream, index))


def configure_determinism(deterministic: bool = True) -> None:
    """Request reproducible PyTorch kernels, shared by training and retrieval.

    Raise if an operation has no deterministic implementation.
    ``deterministic=False`` leaves the backend settings untouched.
    """
    if type(deterministic) is not bool:
        raise ValueError("runtime.deterministic must be true or false")
    if not deterministic:
        return
    # Read when cuBLAS is first used, so it must be set before any CUDA matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device) from one value."""
    configure_determinism(deterministic)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def seed_feature_extraction(cfg: dict) -> None:
    """Start each frozen encoder from the same state, independent of train.seed."""
    seed_everything(0, deterministic=cfg["runtime"].get("deterministic", True))


def numerical_runtime(device) -> dict:
    """Numerical settings that must match when continuing a partial run or cache."""
    device = torch.device(device)
    cuda = device.type == "cuda"
    return {
        "device": str(device),
        "torch": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cuda
        else None,
    }


@contextmanager
def evaluating(*modules):
    """Temporarily switch modules to eval mode and restore their state."""
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.train(mode)
