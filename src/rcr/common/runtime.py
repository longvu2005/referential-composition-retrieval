"""Device selection shared by feature extraction and retrieval methods."""

import torch


def resolve_device(cfg):
    name = cfg["runtime"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)
