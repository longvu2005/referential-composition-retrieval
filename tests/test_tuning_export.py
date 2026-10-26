"""Full-gallery exports must replay new shortlists without a model or cache."""

import copy
import json
from pathlib import Path

import pytest
import torch

from rcr.common.io import save_results, sha256_file
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed import calibration, retrieve
from rcr.proposed.ranking import TUNING_CONVENTIONS
from rcr.proposed.replay import main as replay_main
from rcr.proposed.replay import replay_rankings
from tests.test_ranking import _rank
from tests.test_ranking import tiny_ranker as tiny_ranker
from tests.test_score_pipeline import saved_pipeline as saved_pipeline


def test_export_scores_every_nonself_pair_once_and_retains_gallery_alignment(
    tiny_ranker, monkeypatch
):
    cache, model, _, samples, _ = tiny_ranker
    active, pairs = [], []
    encode, score = model.encode_query, model.score_target

    def image_index(scene):
        return (cache.scenes == scene).flatten(1).all(1).nonzero().item()

    def encode_counted(scene, *args, **kwargs):
        active[:] = [image_index(scene[0])]
        return encode(scene, *args, **kwargs)

    def score_counted(query, scene, *args, **kwargs):
        pairs.extend((active[0], image_index(row)) for row in scene)
        return score(query, scene, *args, **kwargs)

    monkeypatch.setattr(model, "encode_query", encode_counted)
    monkeypatch.setattr(model, "score_target", score_counted)
    output = _rank(tiny_ranker, top_m=1, coarse_beta=0.0, save_tuning=True)
    tuning = output["tuning_scores"]
    assert all(tuning[key] == value for key, value in TUNING_CONVENTIONS.items())
    expected = {
        (cache.by_id[sample["query_image_id"]], j)
        for sample in samples
        for j in range(len(cache.image_ids))
        if cache.image_ids[j] != sample["query_image_id"]
    }
    assert len(pairs) == len(set(pairs)) == len(expected)
    assert set(pairs) == expected
    assert (
        tuning["state_scores"] is not None
    )  # Saved even when the original beta is zero.
    top = output["coarse_topm"].long()
    for key in (
        "identity_scores",
        "state_scores",
        "fafa_binding",
        "context",
        "coverage",
    ):
        assert tuning[key].shape == (len(samples), len(cache.image_ids))
        assert tuning[key].dtype == torch.float32
        assert tuning[key].device.type == "cpu"
    for key in ("fafa_binding", "context", "coverage"):
        assert torch.equal(tuning[key].gather(1, top), output[key])
        assert (
            tuning[key][
                torch.arange(len(samples)), tuning["query_indices"]
            ].count_nonzero()
            == 0
        )
    assert {
        "case_type",
        "identity_ids",
        "positive_image_ids",
        "target_image_id",
    }.isdisjoint(tuning)
    assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


@pytest.mark.parametrize("mode", ["context_zscore", "legacy_raw"])
@pytest.mark.parametrize("beta", [0.0, 0.4, 2.0])
@pytest.mark.parametrize("top_m", [1, 3, 20])
def test_new_beta_topm_and_fine_weights_match_direct_retrieval(
    tiny_ranker, mode, beta, top_m
):
    source = _rank(tiny_ranker, top_m=1, coarse_beta=0.0, save_tuning=True)
    replayed = replay_rankings(
        source,
        coarse_beta=beta,
        top_m=top_m,
        fine_score_mode=mode,
        context_weight=0.13,
        fine_coarse_weight=0.7,
    )
    direct = _rank(
        tiny_ranker,
        coarse_beta=beta,
        top_m=top_m,
        fine_score_mode=mode,
        context_weight=0.13,
        fine_coarse_weight=0.7,
    )
    for key in (
        "coarse_rankings",
        "coarse_topm",
        "fine_rankings",
        "fafa_rankings",
        "rankings",
    ):
        assert torch.equal(replayed[key], direct[key]), key
    for key in (
        "coarse_topm_scores",
        "fafa_binding",
        "context",
        "coverage",
        "fine_scores",
    ):
        torch.testing.assert_close(replayed[key], direct[key], atol=2e-6, rtol=2e-5)
    if top_m > 1:
        assert replayed["coarse_topm"].shape[1] > source["coarse_topm"].shape[1]
    assert replayed["tuning_scores"] is source["tuning_scores"]
    assert source["score_config"]["coarse_beta"] == 0.0
    assert source["score_config"]["top_m"] == 1
    # Changing Top-M twice must retain evidence outside both intermediate shortlists.
    again = replay_rankings(replayed, coarse_beta=0.4, top_m=4)
    fresh = replay_rankings(source, coarse_beta=0.4, top_m=4)
    assert torch.equal(again["rankings"], fresh["rankings"])


@pytest.mark.parametrize("override", [{"coarse_beta": 1.0}, {"top_m": 1}])
def test_schema1_shortlist_cannot_claim_new_coarse_or_topm(tiny_ranker, override):
    source = _rank(tiny_ranker)
    with pytest.raises(ValueError, match="full-gallery --save-tuning"):
        replay_rankings(source, **override)
    # Existing callers can still request the original values explicitly.
    assert torch.equal(
        replay_rankings(source, coarse_beta=0.4, top_m=4)["rankings"],
        source["rankings"],
    )


@pytest.mark.parametrize(
    "field,mutate,message",
    [
        ("scope", lambda value: "coarse_topm", "contract"),
        ("identity_scores", lambda value: value[:, :-1], "full gallery"),
        ("context", lambda value: value.half(), "float32"),
        ("coverage", lambda value: value + 0.1, "misaligned"),
        ("state_scores", lambda value: None, "state_scores"),
        ("query_indices", lambda value: value + 100, "query_indices"),
        ("state_trained", lambda value: "yes", "state branch"),
    ],
)
def test_replay_rejects_incomplete_or_misaligned_full_gallery_evidence(
    tiny_ranker, field, mutate, message
):
    source = _rank(tiny_ranker, top_m=1, save_tuning=True)
    source["tuning_scores"][field] = mutate(source["tuning_scores"][field])
    with pytest.raises(ValueError, match=message):
        replay_rankings(source, coarse_beta=2.0)


def test_export_does_not_enable_an_untrained_state_branch(tiny_ranker, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("untrained state features must not be projected")

    monkeypatch.setattr(tiny_ranker[1], "encode_image_state", forbidden)
    source = _rank(tiny_ranker, save_tuning=True, state_trained=False, coarse_beta=0.0)
    assert source["tuning_scores"]["state_scores"] is None
    assert not source["tuning_scores"]["state_trained"]
    assert torch.equal(replay_rankings(source)["rankings"], source["rankings"])
    with pytest.raises(ValueError, match="state branch is untrained"):
        replay_rankings(source, coarse_beta=0.4)


def test_beta_alone_can_admit_a_previously_unscored_shortlist_candidate(tiny_ranker):
    source = copy.deepcopy(
        _rank(tiny_ranker, top_m=1, coarse_beta=0.0, save_tuning=True)
    )
    tuning = source["tuning_scores"]
    old = source["coarse_topm"][0, 0].item()
    query = tuning["query_indices"][0].item()
    new = next(i for i in range(len(source["gallery_ids"])) if i not in (old, query))
    tuning["identity_scores"][0].zero_()
    tuning["identity_scores"][0, old] = 1.0
    tuning["state_scores"][0].zero_()
    tuning["state_scores"][0, new] = 1.0
    original = replay_rankings(source)
    changed = replay_rankings(original, coarse_beta=4.0)
    assert original["coarse_topm"][0, 0] == old
    assert changed["coarse_topm"][0, 0] == new
    for key in ("fafa_binding", "context", "coverage"):
        assert changed[key][0, 0] == tuning[key][0, new]
    assert torch.equal(changed["rankings"][:, 1:], changed["coarse_rankings"][:, 1:])


def test_nonfinite_state_eligibility_is_recomputed_for_each_beta(tiny_ranker):
    source = copy.deepcopy(
        _rank(tiny_ranker, top_m=20, coarse_beta=0.0, save_tuning=True)
    )
    index = source["coarse_topm"][0, 0].item()
    source["tuning_scores"]["state_scores"][0, index] = torch.nan
    zero = replay_rankings(source, coarse_beta=0.0)
    positive = replay_rankings(source, coarse_beta=0.4)
    assert torch.isfinite(zero["coarse_topm_scores"][0]).all()
    position = (positive["coarse_topm"][0] == index).nonzero().item()
    assert torch.isneginf(positive["coarse_topm_scores"][0, position])
    assert torch.isneginf(positive["fused_topm_scores"][0, position])


def test_full_export_save_load_cli_and_metadata_without_checkpoint_or_cache(
    tiny_ranker, tmp_path
):
    source = _rank(tiny_ranker, top_m=1, coarse_beta=0.0, save_tuning=True)
    save_results(
        tmp_path / "source",
        source,
        {
            "split": "val",
            "retrieval": tiny_ranker[4],
            "config": {
                "retrieval": tiny_ranker[4],
                "selected_checkpoint_sha256": "old",
                "selected_validation_sha256": "old",
                "selected_retrieval": {"coarse_beta": 0.0},
            },
        },
    )
    path = tmp_path / "source/rankings.pt"
    digest = sha256_file(path)
    replay_main(
        [
            str(path),
            "--output-dir",
            str(tmp_path / "offline/val"),
            "--coarse-beta",
            "2",
            "--top-m",
            "4",
            "--context-weight",
            "0.13",
            "--fine-coarse-weight",
            "0.7",
        ]
    )
    saved = torch.load(tmp_path / "offline/val/rankings.pt", weights_only=True)
    expected = replay_rankings(
        source, coarse_beta=2.0, top_m=4, context_weight=0.13, fine_coarse_weight=0.7
    )
    assert torch.equal(saved["rankings"], expected["rankings"])
    metadata = json.loads((tmp_path / "offline/val/run.json").read_text())
    assert metadata["retrieval"] == metadata["config"]["retrieval"]
    assert metadata["retrieval"]["coarse_beta"] == 2.0
    assert metadata["retrieval"]["top_m"] == 4
    assert "selected_retrieval" not in metadata["config"]
    assert metadata["replay"]["scope"] == "full_gallery"
    assert metadata["replay"]["source_sha256"] == digest == sha256_file(path)


def test_standalone_export_new_shortlist_replay_matches_official_metrics(
    saved_pipeline,
):
    cfg, data, _ = saved_pipeline
    cfg["retrieval"].update(top_m=1, coarse_beta=0.0)
    cfg["candidate_ks"] = [1]
    exported = retrieve.retrieve(cfg, save_tuning=True)
    directory = Path(cfg["output"]["dir"]) / "val"
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert torch.equal(saved["rankings"], exported["rankings"])
    assert saved["tuning_scores"]["identity_scores"].shape == (1, 3)
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["tuning_export"]["scope"] == "full_gallery"
    assert (
        saved["tuning_scores"]["provenance"]["checkpoint_sha256"]
        == metadata["checkpoint_sha256"]
    )
    assert (
        saved["tuning_scores"]["provenance"]["validation_sha256"]
        == metadata["validation_sha256"]
    )
    replayed = replay_rankings(saved, coarse_beta=1.5, top_m=2, fine_coarse_weight=0.1)
    direct_cfg = copy.deepcopy(cfg)
    direct_cfg["output"]["dir"] = str(directory.parent / "direct")
    direct_cfg["retrieval"].update(coarse_beta=1.5, top_m=2, fine_coarse_weight=0.1)
    direct = retrieve.retrieve(direct_cfg)
    assert torch.equal(replayed["rankings"], direct["rankings"])
    assert evaluate_retrieval_output(
        data, [data.samples[1]], replayed, [1], split="val"
    ) == evaluate_retrieval_output(data, [data.samples[1]], direct, [1], split="val")
    bad = copy.deepcopy(cfg)
    bad["split"] = "test"
    with pytest.raises(ValueError, match="tuning export requires the validation split"):
        retrieve.retrieve(bad, save_tuning=True)


def test_export_rejects_truncated_rankings(tiny_ranker):
    from rcr.proposed.ranking import rank_weight_grid
    from tests.test_train import Tokenizer

    cache, model, text, samples, settings = tiny_ranker
    with pytest.raises(ValueError, match="complete gallery rankings"):
        rank_weight_grid(
            samples,
            cache,
            Tokenizer(),
            text,
            model,
            torch.device("cpu"),
            gallery_ids=cache.image_ids,
            settings=settings,
            weights=[(0.4, 0.4)],
            save_tuning=True,
            ranking_limit=2,
        )


def test_offline_calibration_selects_and_freezes_without_checkpoint_or_caches(
    saved_pipeline, tmp_path, monkeypatch
):
    import yaml

    from rcr.evaluation.report import load_run

    cfg, data, _ = saved_pipeline
    cfg["retrieval"].update(top_m=1, coarse_beta=0.0)
    cfg["candidate_ks"] = [1]
    source = retrieve.retrieve(cfg, save_tuning=True)
    source_path = tmp_path / "scores-only.pt"
    torch.save(source, source_path)
    digest = sha256_file(source_path)
    # The offline path must not need checkpoint bytes, sibling metadata or cache shards.
    offline_cfg = copy.deepcopy(cfg)
    offline_cfg["checkpoint"] = str(tmp_path / "not-present.pt")
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(offline_cfg))

    def forbidden(*args, **kwargs):
        pytest.fail("offline calibration must not run model retrieval")

    monkeypatch.setattr(calibration, "retrieve_weight_grid", forbidden)
    root = tmp_path / "offline-calibration"
    rows = calibration.calibrate(
        {
            "model_config": str(config_path),
            "rankings": str(source_path),
            "output_dir": str(root),
            "coarse_beta": [0.0, 1.5],
            "fine_coarse_weight": [0.0, 0.1],
            "context_weight": [0.03, 0.13],
            "top_m": [1, 2],
        }
    )
    assert len(rows) == 16
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    expected = torch.load(root / "selected/val/rankings.pt", weights_only=True)
    best = max(rows, key=lambda row: row["full_map"])
    for key in ("coarse_beta", "fine_coarse_weight", "context_weight", "top_m"):
        assert selected["retrieval"][key] == best[key] == expected["score_config"][key]
        assert selected["selected_retrieval"][key] == best[key]
    assert (
        selected["selected_checkpoint_sha256"]
        == source["tuning_scores"]["provenance"]["checkpoint_sha256"]
    )
    assert (
        selected["selected_validation_sha256"]
        == source["tuning_scores"]["provenance"]["validation_sha256"]
    )
    assert sha256_file(source_path) == digest
    assert (
        load_run("proposed", root / "selected/val", "val")["metrics"]["overall"][
            "full_map"
        ]
        == best["full_map"]
    )
    assert not list(root.glob("grid_*"))  # Only the best full ranking is serialized.
    selected["checkpoint"] = cfg["checkpoint"]  # The same checkpoint bytes may move.
    selected["output"]["dir"] = str(tmp_path / "direct-selected")
    direct = retrieve.retrieve(selected)
    assert torch.equal(direct["rankings"], expected["rankings"])


@pytest.mark.parametrize("changed", ["query_subset", "validation_sha256"])
def test_offline_calibration_rejects_partial_or_changed_validation_before_writes(
    saved_pipeline, tmp_path, changed
):
    import yaml

    cfg, _, _ = saved_pipeline
    source = retrieve.retrieve(cfg, save_tuning=True)
    source["tuning_scores"]["provenance"][changed] = (
        True if changed == "query_subset" else "changed"
    )
    path = tmp_path / "source.pt"
    torch.save(source, path)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "rejected"
    with pytest.raises(
        ValueError, match="complete validation|changed since tuning export"
    ):
        calibration.calibrate(
            {
                "model_config": str(config_path),
                "rankings": str(path),
                "output_dir": str(root),
                "coarse_beta": [0.0],
                "fine_coarse_weight": [0.0],
            }
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "split,save_tuning", [("val", False), ("val", True), ("test", False)]
)
def test_run_uses_configured_split_and_checkpoint(
    tmp_path, monkeypatch, split, save_tuning
):
    import yaml

    from rcr.proposed import runner

    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["split"] = split
    cfg["checkpoint"] = "frozen/best.pt"
    cfg["selected_checkpoint_sha256"] = "frozen-checkpoint"
    original = copy.deepcopy(cfg)
    calls = []

    def stage(command, **kwargs):
        calls.append(command)
        if command[2] == "evaluate":
            directory = Path(cfg["output"]["dir"]) / split
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "metrics.json").write_text(json.dumps({"overall": {}}))

    monkeypatch.setattr(runner.subprocess, "run", stage)
    rows = runner.run_experiment(cfg, Path("tools/run.py"), save_tuning=save_tuning)
    assert [cmd[2] for cmd in calls] == ["retrieve", "evaluate"]
    for command in calls:
        assert ("--save-tuning" in command) == (
            save_tuning and command[2] == "retrieve"
        )
        saved = yaml.safe_load(Path(command[4]).read_text())
        assert saved == original
    assert cfg == original
    assert rows == [
        {"method": "proposed", "split": split, "checkpoint": "frozen/best.pt"}
    ]


def test_misspelled_saved_score_setting_cannot_start_model_calibration():
    with pytest.raises(ValueError, match="unknown calibration settings: ranking"):
        calibration.calibrate({"ranking": "scores.pt"})
