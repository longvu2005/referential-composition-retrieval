"""Slow partial-transport regimes, independent optima and derivatives."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from rcr.proposed.cache import build as builder
from rcr.proposed.nn.matching import TransportConvergenceError, partial_transport
from tests.proposed.test_text_and_pipeline import experiment as _experiment
from tests.proposed.test_text_and_pipeline import tiny_clip as _tiny_clip

experiment = _experiment
tiny_clip = _tiny_clip


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(42)


@pytest.mark.parametrize(
    "q,t,total,null", [(3, 2, 2.0001, -2), (3, 2, 2.0001, -5), (6, 1, 1.002, -1.5)]
)
def test_nearly_full_capacities_converge_without_changing_objective(q, t, total, null):
    # Equal scores/masses have a closed-form optimum. The original alternating
    # solver fails to reach 1e-5 after 512 updates in these near-balanced cases.
    logits = torch.zeros(1, q, t + 1, requires_grad=True)
    with torch.no_grad():
        logits[..., 0] = null
    mass = torch.full((1, q), total / q)
    x, diagnostics = partial_transport(logits, mass, torch.ones(1, t, dtype=torch.bool))
    expected = torch.full_like(x, 1 / q)
    expected[..., 0] = mass - t / q
    torch.testing.assert_close(x, expected, atol=1e-5, rtol=1e-5)
    assert diagnostics["iterations"] < 64
    assert max(diagnostics[k] for k in diagnostics if k != "iterations") <= 1e-5
    (x * torch.randn_like(x)).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_asymmetric_nearly_balanced_optimum_matches_slsqp():
    logits = torch.tensor(
        [[[-2, -0.05, -0.1], [-2, -0.08, -0.02], [-2, -0.03, -0.04]]],
        dtype=torch.float64,
    )
    mass = torch.tensor([[0.68, 0.65, 0.6701]], dtype=torch.float64)
    x, _ = partial_transport(
        logits, mass, torch.ones(1, 2, dtype=torch.bool), tolerance=1e-10
    )
    ell, weights = logits.numpy()[0], mass.numpy()[0]

    def objective(flat):
        y = flat.reshape(3, 3)
        return -(y * ell).sum() + 0.2 * (y * np.log(y)).sum()

    total = weights.sum()
    initial = np.tile((weights / total)[:, None], (1, 3))
    initial[:, 0] *= total - 2
    solved = minimize(
        objective,
        initial.flatten(),
        jac=lambda flat: (-ell + 0.2 * (1 + np.log(flat.reshape(3, 3)))).flatten(),
        bounds=[(1e-12, None)] * 9,
        constraints=[
            {"type": "eq", "fun": lambda flat: flat.reshape(3, 3).sum(1) - weights},
            {"type": "ineq", "fun": lambda flat: 1 - flat.reshape(3, 3)[:, 1:].sum(0)},
        ],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert solved.success, solved.message
    np.testing.assert_allclose(x.numpy().reshape(-1), solved.x, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("nearly_balanced", [False, True])
def test_active_capacity_gradients_for_logits_and_mass(nearly_balanced):
    logits = torch.tensor(
        [[[-0.8, -0.2, -0.5], [-0.8, -0.3, -0.6], [-0.8, -0.1, -0.2]]],
        dtype=torch.float64,
    )
    mass = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float64)
    if nearly_balanced:
        logits[..., 0] = -3
        mass = torch.tensor([[0.68, 0.65, 0.6701]], dtype=torch.float64)
    logits.requires_grad_()
    mass.requires_grad_()
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda ell, w: partial_transport(ell, w, mask, tolerance=1e-12)[0],
        (logits, mass),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-3,
    )


@pytest.mark.parametrize("seed", [0, 3, 9])
@pytest.mark.parametrize("scale", [1, 10, 30])
def test_heterogeneous_batched_transport_is_feasible_and_differentiable(seed, scale):
    torch.manual_seed(seed)
    logits = (scale * torch.randn(8, 24, 9)).requires_grad_()
    mass = torch.rand(8, 24, requires_grad=True)
    mask = torch.rand(8, 8) > 0.3
    mask[0] = False
    with torch.no_grad():
        mass[1] = 0
    x, diagnostics = partial_transport(logits, mass, mask)
    torch.testing.assert_close(x.sum(-1), mass, atol=1e-5, rtol=1e-5)
    assert (x[..., 1:].sum(1) <= 1 + 1e-5).all()
    assert diagnostics["dual_residual"] <= 1e-5
    assert x[..., 1:].masked_select(~mask[:, None]).count_nonzero() == 0
    grads = torch.autograd.grad((x * torch.randn_like(x)).sum(), (logits, mass))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[0][..., 1:].masked_select(~mask[:, None]).count_nonzero() == 0


def test_training_records_reproducible_transport_failure(experiment, monkeypatch):
    cfg, _ = experiment
    builder.prepare_cache(cfg)
    training = importlib.import_module("rcr.proposed.train")

    def fail_transport(*args, **kwargs):
        return partial_transport(
            torch.tensor([[[-5.0, 0.0]]]).expand(1, 3, 2),
            torch.ones(1, 3),
            torch.ones(1, 1, dtype=torch.bool),
            max_iterations=1,
        )

    monkeypatch.setattr(training, "compute_loss", fail_transport)
    with pytest.raises(TransportConvergenceError, match="did not converge"):
        training.train(cfg)
    output = Path(cfg["output"]["dir"])
    report = json.loads((output / "transport_batch.json").read_text())
    assert report["epoch"] == 1 and report["sample_ids"] == ["train"]
    assert report["max_iterations"] == report["iterations"] == 1
    assert not (output / "last.pt").exists() and not (output / "best.pt").exists()
    x, _ = partial_transport(
        torch.tensor(report["logits"]),
        torch.tensor(report["mass"]),
        torch.tensor(report["real_mask"]),
        tau=report["tau"],
        tolerance=report["tolerance"],
    )
    torch.testing.assert_close(x.sum(-1), torch.ones(1, 3), atol=1e-5, rtol=1e-5)
