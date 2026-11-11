"""Keep process-wide determinism switches from leaking between tests.

Training turns on PyTorch's reproducible-kernel mode for the whole process. A
test that trains must not change how every later test runs.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_determinism_switches():
    try:
        import torch
    except ImportError:  # Annotation-only environments have no Torch.
        yield
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
