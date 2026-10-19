"""Small runtime helpers shared by training and retrieval."""

from contextlib import contextmanager

import torch


def resolve_device(cfg):
    name = cfg["runtime"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


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
