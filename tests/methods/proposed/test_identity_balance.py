"""Controlled identity ablations: math, initialization, checkpoints and protocol."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from rcr.methods.common.data import load_rcr_data
from rcr.methods.proposed import ablation, inference
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.identity_balance import (
    BranchNorms,
    IdentityBalance,
    parameter_fingerprint,
)
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.shortlist import load_fixed_coarse, validate_fixed_coarse
from tests.methods.proposed.test_runner import experiment as experiment
from tests.methods.proposed.test_runner import local_stages as local_stages
from tools.methods import run as cli


@pytest.mark.parametrize("mode", ["none", "pre_scale", "layernorm", "l2"])
def test_ablation_adds_no_parameters_or_rng_draws(mode):
    torch.manual_seed(19)
    control = RCRModel(8, 6, 2)
    rng = torch.get_rng_state()
    torch.manual_seed(19)
    variant = RCRModel(8, 6, 2, identity_balance={"mode": mode, "scale": 8})
    assert torch.equal(rng, torch.get_rng_state())
    assert parameter_fingerprint(control) == parameter_fingerprint(variant)
    assert not list(variant.composition.identity_balance.parameters())
    assert not list(variant.target_builder.identity_balance.parameters())


def test_pre_scale_keeps_projection_bias_unscaled():
    projection = nn.Linear(2, 3)
    with torch.no_grad():
        projection.weight.fill_(0.2)
        projection.bias.fill_(0.7)
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    balance = IdentityBalance({"mode": "pre_scale", "scale": 8})
    y = balance(x, projection)
    torch.testing.assert_close(y, F.linear(8 * x, projection.weight, projection.bias))
    assert not torch.allclose(y, 8 * projection(x))
    y.sum().backward()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "mode,alpha", [("layernorm", 1), ("l2", 0.5), ("l2", 1), ("l2", 2)]
)
def test_radius_direction_and_gradient_survive_serialization(mode, alpha):
    torch.manual_seed(7)
    layer = nn.Linear(3, 8)
    x = torch.randn(2, 4, 3, requires_grad=True)
    balance = IdentityBalance({"mode": mode, "alpha": alpha})
    with pytest.raises(ValueError, match="uninitialized"):
        balance(x, layer)
    balance.radius.fill_(3.0)
    value = balance(x, layer)
    torch.testing.assert_close(value.norm(dim=-1), torch.full((2, 4), 3.0 * alpha))
    if mode == "l2":
        torch.testing.assert_close(
            F.cosine_similarity(value, layer(x), dim=-1), torch.ones(2, 4)
        )
    else:
        torch.testing.assert_close(
            value.mean(dim=-1), torch.zeros(2, 4), atol=1e-6, rtol=0
        )
    value[..., 0].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    radius = balance.radius.clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    optimizer.step()
    assert torch.equal(balance.radius, radius)
    restored = IdentityBalance({"mode": mode, "alpha": alpha})
    restored.load_state_dict(balance.state_dict())
    torch.testing.assert_close(restored(x, layer), balance(x, layer))


def test_norm_moments_are_token_weighted_and_ignore_padding():
    norms = BranchNorms()
    norms.add(
        "x",
        torch.tensor([[3.0, 4.0], [float("nan"), 0.0]]),
        torch.tensor([True, False]),
    )
    norms.add("x", torch.tensor([[0.0, 0.0], [0.0, 10.0]]))
    assert norms.report()["x"]["count"] == 3
    assert norms.report()["x"]["mean"] == 5.0


def tiny_suite(path, cfg, root):
    suite = yaml.safe_load(Path("configs/ablations/identity_balance.yaml").read_text())
    suite.update(base_config=str(path), output_dir=str(root))
    suite["fixed_coarse"]["checkpoint"] = cfg["checkpoint"]
    suite["overrides"].update({"retrieval.top_m": 2, "candidate_ks": [1, 2]})
    return suite


def test_eight_variants_share_initialization_radii_and_real_shortlists(
    experiment,
    local_stages,
    tmp_path,
    monkeypatch,
):
    cfg, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    source = Path(cfg["checkpoint"]).read_bytes()
    root = tmp_path / "identity_suite"
    suite = tiny_suite(path, cfg, root)
    expanded = ablation.expand_experiments(suite)
    assert len(expanded) == 8
    for e in expanded:
        assert e["config"]["loss"] == cfg["loss"]
        assert e["config"]["train"] == cfg["train"]
    # Force a known validation winner independently of tiny tied image fixtures.
    # All rankings, checkpoints, metrics, training and frozen artifacts are real.
    run = ablation.run_experiment

    def controlled(config, entrypoint, **kwargs):
        results = run(config, entrypoint, **kwargs)
        if kwargs.get("train"):
            metrics_path = Path(config["output"]["dir"]) / "val/metrics.json"
            metrics = json.loads(metrics_path.read_text())
            mode = config["model"]["identity_balance"]["mode"]
            metrics["overall"]["full_map"] = 0.8 if mode == "layernorm" else 0.2
            metrics_path.write_text(json.dumps(metrics))
        return results

    monkeypatch.setattr(ablation, "run_experiment", controlled)
    rows = ablation.run_ablation(suite, Path(cli.__file__), splits=["val", "test"])
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(val) == 8 and {r["experiment"] for r in test} == {
        "A_control",
        "C_layernorm",
    }
    assert len({r["initialization_sha256"] for r in val}) == 1
    assert len({(r["R_q"], r["R_t"]) for r in val}) == 1
    assert all(r["R_q"] > 0 and r["R_t"] > 0 for r in val)
    assert all("fine_id_r1" in r and "norm_target.evidence_geometry" in r for r in val)
    frozen = torch.load(root / "fixed_coarse/val/rankings.pt", weights_only=True)
    for row in val:
        run_root = root / row["experiment"]
        output = torch.load(run_root / "val/rankings.pt", weights_only=True)
        assert torch.equal(output["coarse_rankings"], frozen["coarse_rankings"])
        assert torch.equal(output["coarse_topm"], frozen["coarse_topm"])
        checkpoint = torch.load(run_root / "best.pt", weights_only=True)
        initial = json.loads((run_root / "initialization.json").read_text())
        calibration = checkpoint["identity_calibration"]
        assert calibration == initial["identity_calibration"]
        assert calibration["sample_ids"] == ["train"]
        assert checkpoint["model"][
            "composition.identity_balance.radius"
        ].item() == pytest.approx(row["R_q"])
        periodic = json.loads(
            (run_root / "evaluation/epoch_001/val_metrics.json").read_text()
        )
        actual = json.loads((run_root / "val/metrics.json").read_text())
        assert periodic["overall"]["fine_id_r1"] == actual["overall"]["fine_id_r1"]
    assert Path(cfg["checkpoint"]).read_bytes() == source
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    assert selected["model"]["identity_balance"]["mode"] == "layernorm"
    followup = yaml.safe_load((root / "confirmation.yaml").read_text())
    followups = ablation.expand_experiments(followup)
    assert len(followups) == 4
    assert {e["config"]["train"]["seed"] for e in followups} == {1, 2}
    assert {e["config"]["model"]["identity_balance"]["mode"] for e in followups} == {
        "none",
        "layernorm",
    }
    assert all(
        e["config"]["evaluation"]["fixed_coarse"]
        == selected["evaluation"]["fixed_coarse"]
        for e in followups
    )
    # Run the ordinary generated suite; no third training implementation.
    more = ablation.run_ablation(followup, Path(cli.__file__))
    assert len(more) == 4
    for seed in (1, 2):
        pair = [r for r in more if r["seed"] == seed]
        assert pair[0]["initialization_sha256"] == pair[1]["initialization_sha256"]
        assert (pair[0]["R_q"], pair[0]["R_t"]) == (pair[1]["R_q"], pair[1]["R_t"])
    # Replaying selected.yaml loads stored radii; no training calibration in inference.
    inference.retrieve(selected)
    artifact = Path(selected["evaluation"]["fixed_coarse"]["val"]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    data = load_rcr_data(cfg["data"]["final_dir"])
    cache = GalleryCache(cfg["data"]["cache"])
    with pytest.raises(ValueError, match="artifact changed"):
        load_fixed_coarse(selected, data, cache.cache_id, "val")


def test_fixed_coarse_rejects_reordered_gallery_and_invalid_scores():
    output = {
        "gallery_ids": ["q", "a", "b"],
        "sample_ids": ["s"],
        "coarse_rankings": torch.tensor([[1, 2]]),
        "coarse_scores": torch.tensor([[-torch.inf, 2.0, 1.0]]),
    }
    samples = [{"sample_id": "s", "query_image_id": "q"}]
    assert validate_fixed_coarse(output, samples, output["gallery_ids"]) == {"s": 0}
    with pytest.raises(ValueError, match="gallery order"):
        validate_fixed_coarse(output, samples, ["q", "b", "a"])
    corrupt = copy.deepcopy(output)
    corrupt["coarse_rankings"][0, 0] = 2
    with pytest.raises(ValueError, match="permutation"):
        validate_fixed_coarse(corrupt, samples, output["gallery_ids"])
    corrupt = copy.deepcopy(output)
    corrupt["coarse_scores"][0, 2] = 3
    with pytest.raises(ValueError, match="disagree"):
        validate_fixed_coarse(corrupt, samples, output["gallery_ids"])


def test_legacy_checkpoint_without_balance_config_still_loads(experiment, local_stages):
    cfg, path = experiment
    cli.main(
        ["run", "--config", str(path), "--build-cache", "--train", "--splits", "val"]
    )
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    del checkpoint["config"]["model"]["identity_balance"]
    checkpoint["model"] = {
        k: v for k, v in checkpoint["model"].items() if ".identity_balance." not in k
    }
    torch.save(checkpoint, cfg["checkpoint"])
    result = inference.retrieve(cfg)
    assert result["rankings"].shape == (1, 2)


def test_calibration_preserves_rng_parameters_modes_and_removes_observers(
    experiment,
    local_stages,
):
    from rcr.methods.common.data import split_samples
    from rcr.methods.proposed.encoders import TextEncoder
    from rcr.methods.proposed.identity_balance import calibrate_identity
    from tests.methods.proposed.test_train_cli import Backbone, Tokenizer

    cfg, path = experiment
    cli.main(["build-cache", "--config", str(path)])
    data = load_rcr_data(cfg["data"]["final_dir"])
    cache = GalleryCache(cfg["data"]["cache"])
    dim = cache.persons.shape[-1]
    model = RCRModel(dim, 6, 2, identity_balance={"mode": "l2"}).train()
    encoder = TextEncoder(Backbone(), dim).train()
    before, rng = parameter_fingerprint(model, encoder), torch.get_rng_state()
    result = calibrate_identity(
        model,
        split_samples(data, "train"),
        cache,
        Tokenizer(),
        encoder,
        torch.device("cpu"),
        max_queries=1,
    )
    assert result["sample_ids"] == ["train"]
    assert parameter_fingerprint(model, encoder) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training and encoder.training
    assert model.composition.identity_balance.mode == "l2"
    assert model.composition.norm_observer is model.target_builder.norm_observer is None


def test_dual_fine_only_and_pipeline_metrics_use_distinct_full_rankings():
    from rcr.evaluation.evaluate import evaluate_retrieval_output
    from tests.evaluation.test_saved_results import inputs

    data, samples = inputs()
    samples[0]["case_type"] = "DUAL"
    samples[0]["subjects"].append({"subject_id": 2, "identity_ids": ["p2"]})
    for image in ("q", "a"):
        data.gt_head_boxes_by_image[image].append({"identity_id": "p2"})
    output = {
        "sample_ids": ["s"],
        "gallery_ids": ["q", "a", "b"],
        "rankings": torch.tensor([[1, 2]]),
        "fine_rankings": torch.tensor([[2, 1]]),
        "coarse_rankings": torch.tensor([[1, 2]]),
        "coarse_topm": torch.tensor([[1, 2]]),
    }
    result = evaluate_retrieval_output(data, samples, output, [2], split="test")
    dual = result["by_case"]["DUAL"]
    assert dual["full_map"] == dual["full_r1"] == 1.0
    assert dual["fine_id_r1"] == dual["fine_full_r1"] == 0.0
    assert dual["fine_full_map"] == 0.5
    assert result["overall"]["fine_id_r1"] == 0.0
