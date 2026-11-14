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
    return torch.device(name)


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

    An operation that has no deterministic implementation only warns, so a run is
    never aborted. ``deterministic=False`` leaves the backend settings untouched.
    """
    if not deterministic:
        return
    # Read when cuBLAS is first used, so it must be set before any CUDA matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device) from one value."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    configure_determinism(deterministic)


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
