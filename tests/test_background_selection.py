"""Background selection must follow the same model through train/val/test."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.config import load_config
from rcr.evaluation.runner import evaluate_run
from rcr.proposed import calibration, retrieve, runner
from rcr.proposed.contracts import SELECTED_RETRIEVAL_KEYS
from rcr.proposed.losses import compute_loss, loss_settings
from rcr.proposed.matching import exclusive_membership, matching_settings
from rcr.proposed.replay import replay_rankings
from rcr.proposed.scores import fafa_binding_scores, identity_scores
from tests.test_score_pipeline import saved_pipeline as saved_pipeline
from tests.test_train import _sample
from tests.test_train import objective_batch as objective_batch


@pytest.mark.parametrize("background", [0.0, -3.5])
def test_background_softmax_matches_formula_and_legacy_default(background):
    logits = torch.tensor(
        [[[-2.0, 1.0, 4.0, 0.0], [-1.0, 3.0, 1.0, 2.0]]], requires_grad=True
    )
    subjects = torch.tensor([[True, True]])
    mask = torch.tensor([[True, True, True, False]])
    before = logits.detach().clone()
    membership = exclusive_membership(
        logits, subjects, mask, background_logit=background
    )
    p = torch.cat((torch.full((1, 1, 4), background), logits.detach()), 1).softmax(1)
    confidence = 1 - p[:, 0]
    best, owner = p[:, 1:].max(1)
    selected = mask & (confidence >= 0.5) & (best > p[:, 0])
    expected = (
        torch.nn.functional.one_hot(owner, 2).permute(0, 2, 1)
        * (confidence * selected)[:, None]
    )
    torch.testing.assert_close(membership, expected, rtol=0, atol=0)
    assert not membership.requires_grad
    assert torch.equal(logits, before)
    if background == 0:
        assert torch.equal(membership, exclusive_membership(logits, subjects, mask))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_background_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        matching_settings({"background_logit": value})
    with pytest.raises(ValueError, match="finite"):
        exclusive_membership(
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            background_logit=value,
        )
    assert matching_settings({})["background_logit"] == 0


def test_background_changes_fine_path_without_changing_raw_bce_or_coarse(
    objective_batch,
    monkeypatch,
):
    model, batch = objective_batch
    # The training fixture forbids FDA; this test also exercises fine inference.
    monkeypatch.setattr(
        "rcr.proposed.nn.model.fafa_binding_scores", fafa_binding_scores
    )
    with torch.no_grad():
        model.grounding.score.weight.zero_()
        model.grounding.score.bias.fill_(-1)
    weights = deepcopy(model.state_dict())
    outputs = []
    for background in (0.0, -3.5):
        model.fine_matching["background_logit"] = background
        query = model.encode_query(
            batch["query_scene"],
            batch["query_persons"],
            batch["query_boxes"],
            batch["selections"],
            batch["change"],
            batch["subject_pos"],
            (2, 2),
            selection_mask=batch["selection_mask"],
            change_mask=batch["change_mask"],
            query_person_mask=batch["query_person_mask"],
            subject_mask=batch.get("subject_mask"),
            subject_token_mask=batch["subject_token_mask"],
            subject_ids=batch["subject_ids"],
            fafa_query=torch.ones(1, 1, 1, 5),
        )
        coarse = identity_scores(
            query.identity[0],
            query.logits[0],
            model.encode_identity(batch["target_persons"])[0],
            batch["target_mask"][0],
            query_mask=batch["query_person_mask"][0],
        )
        fine = model.score_target(
            query.repeat_candidates(3),
            batch["target_scene"].flatten(0, 1),
            batch["target_persons"].flatten(0, 1),
            batch["target_boxes"].flatten(0, 1),
            (2, 2),
            batch["target_mask"].flatten(0, 1),
            fafa_targets=torch.ones(3, 1, 3, 5),
            return_details=True,
        )
        _, parts = compute_loss(model, batch, (2, 2), **loss_settings({}))
        outputs.append((query, coarse, fine, parts))
    old, new = outputs
    torch.testing.assert_close(old[0].logits, new[0].logits, rtol=0, atol=0)
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    for key in ("grounding", "coarse", "identity", "state"):
        torch.testing.assert_close(old[3][key], new[3][key], rtol=0, atol=0)
    assert old[0].membership.count_nonzero() == 0 < new[0].membership.count_nonzero()
    assert not torch.equal(old[0].reference, new[0].reference)
    assert old[2]["coverage"].sum() == 0 < new[2]["coverage"].sum()
    assert not torch.equal(old[2]["binding"], new[2]["binding"])
    assert model.state_dict().keys() == weights.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key])


@pytest.mark.parametrize("background", [None, 0.0, -3.5])
def test_inference_restores_checkpoint_background_and_records_effective_config(
    saved_pipeline, background
):
    cfg, _, _ = saved_pipeline
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    matching = checkpoint["config"]["model"]["fine_matching"]
    if background is None:
        matching.pop("background_logit")  # A genuinely old checkpoint.
    else:
        matching["background_logit"] = background
    torch.save(checkpoint, cfg["checkpoint"])
    cfg["model"]["fine_matching"]["background_logit"] = 12  # Not an override.
    output = retrieve.retrieve(cfg)
    effective = 0 if background is None else background
    assert output["fine_matching"]["background_logit"] == effective
    provenance = output["provenance"]
    assert provenance["fine_matching"]["background_logit"] == effective
    assert (
        provenance["config"]["model"]["fine_matching"]["background_logit"] == effective
    )
    assert cfg["model"]["fine_matching"]["background_logit"] == 12
    with pytest.raises(TypeError, match="background_logit"):
        replay_rankings(output, background_logit=-3.5)


def test_validation_freezes_background_for_test_and_run(
    saved_pipeline, tmp_path, monkeypatch
):
    cfg, data, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    root = tmp_path / "selection"
    calibration.calibrate(
        {
            "rankings": str(Path(cfg["output"]["dir"]) / "val/rankings.pt"),
            "output_dir": str(root),
            "coarse_beta": [cfg["retrieval"]["coarse_beta"]],
            "context_weight": [cfg["retrieval"]["context_weight"]],
            "fine_coarse_weight": [cfg["retrieval"]["fine_coarse_weight"]],
            "top_m": [2],
        }
    )
    selected = load_config(root / "selected.yaml")
    assert (
        selected["selected_background_logit"]
        == source["fine_matching"]["background_logit"]
        == -3.5
    )
    selected["split"] = "test"
    data.samples[0] = _sample("test", "q", "a")
    data.samples_by_id = {s["sample_id"]: s for s in data.samples}
    data.splits.update(train=[], test=["test"])
    for image in ("q", "a", "n"):
        data.images_by_id[image]["path"] = f"test/{image}.jpg"
    output = retrieve.retrieve(selected)
    assert output["fine_matching"]["background_logit"] == -3.5
    for change in (0.0, None):
        tampered = deepcopy(selected)
        if change is None:
            tampered.pop("selected_background_logit")
        else:
            tampered["selected_background_logit"] = change
        with pytest.raises(ValueError, match="background changed since val selection"):
            retrieve.retrieve(tampered)
    stages = []

    def dispatch(argv, check):
        assert check is True
        effective = yaml.safe_load(Path(argv[4]).read_text())
        stages.append(argv[2])
        if argv[2] == "retrieve":
            result = retrieve.retrieve(effective)
            assert result["fine_matching"]["background_logit"] == -3.5
        else:
            evaluate_run(effective, data=data)

    monkeypatch.setattr(runner.subprocess, "run", dispatch)
    runner.run_experiment(selected, Path("tools/run.py"))
    assert stages == ["retrieve", "evaluate"]


def test_legacy_checkpoint_and_validation_selection_default_to_zero(saved_pipeline):
    cfg, _, _ = saved_pipeline
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    checkpoint["config"]["model"]["fine_matching"].pop("background_logit")
    torch.save(checkpoint, cfg["checkpoint"])
    output = retrieve.retrieve(cfg)
    for key in ("checkpoint_sha256", "validation_sha256"):
        cfg[f"selected_{key}"] = output["provenance"][key]
    cfg["selected_retrieval"] = {
        key: output["score_config"][key] for key in SELECTED_RETRIEVAL_KEYS
    }
    assert "selected_background_logit" not in cfg
    assert retrieve.retrieve(cfg)["fine_matching"]["background_logit"] == 0
