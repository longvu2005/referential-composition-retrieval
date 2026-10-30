"""The sole validation artifact must support model-free calibration and replay."""

import copy
import json
from itertools import product
from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.io import save_results, sha256_file
from rcr.evaluation.evaluate import evaluate_retrieval_output
from rcr.proposed import calibration, replay, retrieve
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import SCORE_SCHEMA_VERSION
from rcr.proposed.replay import main as replay_main
from rcr.proposed.replay import replay_rankings
from tests.test_ranking import _rank
from tests.test_ranking import tiny_ranker as tiny_ranker
from tests.test_score_pipeline import saved_pipeline as saved_pipeline
from tests.test_train import _sample


def _settings(source_path, root):
    return {
        "rankings": str(source_path),
        "output_dir": str(root),
        "coarse_beta": [0.0, 1.5],
        "fine_coarse_weight": [0.0, 0.1],
        "context_weight": [0.03, 0.13],
        "top_m": [1, 2],
    }


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
    output = _rank(tiny_ranker, top_m=1, coarse_beta=0.0)
    assert (
        output["schema_version"]
        == output["score_schema_version"]
        == SCORE_SCHEMA_VERSION
        == 1
    )
    assert output["score_config"]["score_alignment"] == "gallery_ids"
    assert "tuning_scores" not in output
    expected = {
        (cache.by_id[sample["query_image_id"]], j)
        for sample in samples
        for j in range(len(cache.image_ids))
        if cache.image_ids[j] != sample["query_image_id"]
    }
    assert len(pairs) == len(set(pairs)) == len(expected)
    assert set(pairs) == expected
    for key in (
        "identity_scores",
        "state_scores",
        "fafa_binding",
        "context",
        "coverage",
    ):
        assert output[key].shape == (len(samples), len(cache.image_ids))
        assert output[key].dtype == torch.float32
        assert output[key].device.type == "cpu"
    top = output["coarse_topm"].long()
    binding, context, coverage = [
        output[key].gather(1, top) for key in ("fafa_binding", "context", "coverage")
    ]
    torch.testing.assert_close(
        output["fine_raw_scores"],
        output["binding_scale"] * binding + coverage * context,
        rtol=0,
        atol=0,
    )
    for key in ("fafa_binding", "context", "coverage"):
        assert (
            output[key][
                torch.arange(len(samples)), output["query_indices"]
            ].count_nonzero()
            == 0
        )
    assert {
        "case_type",
        "identity_ids",
        "positive_image_ids",
        "target_image_id",
    }.isdisjoint(output)
    assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


@pytest.mark.parametrize("mode", ["context_zscore", "legacy_raw"])
@pytest.mark.parametrize("beta", [0.0, 0.4, 2.0])
@pytest.mark.parametrize("top_m", [1, 3, 20])
def test_new_beta_topm_and_fine_weights_match_direct_retrieval(
    tiny_ranker, mode, beta, top_m
):
    source = _rank(tiny_ranker, top_m=1, coarse_beta=0.0)
    parameters = dict(
        coarse_beta=beta,
        top_m=top_m,
        fine_score_mode=mode,
        context_weight=0.13,
        fine_coarse_weight=0.7,
    )
    replayed = replay_rankings(source, **parameters)
    direct = _rank(tiny_ranker, full_gallery_scores=False, **parameters)
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
        "fine_scores",
        "fine_raw_scores",
        "context_mean",
        "context_std",
        "fused_topm_scores",
    ):
        torch.testing.assert_close(replayed[key], direct[key], atol=2e-6, rtol=2e-5)
    for key in ("fafa_binding", "context", "coverage"):
        torch.testing.assert_close(
            replayed[key].gather(1, replayed["coarse_topm"].long()),
            direct[key],
            atol=2e-6,
            rtol=2e-5,
        )
        assert replayed[key] is source[key]
    assert source["score_config"]["coarse_beta"] == 0.0
    assert source["score_config"]["top_m"] == 1
    again = replay_rankings(replayed, coarse_beta=0.4, top_m=4)
    fresh = replay_rankings(source, coarse_beta=0.4, top_m=4)
    assert torch.equal(again["rankings"], fresh["rankings"])


@pytest.mark.parametrize(
    "field,mutate,message",
    [
        ("identity_scores", lambda value: value[:, :-1], "full gallery"),
        ("context", lambda value: value.half(), "float32"),
        ("coverage", lambda value: value + 0.1, "zero sentinel"),
        ("state_scores", lambda value: None, "state_scores"),
        ("query_indices", lambda value: value + 100, "query_indices"),
        ("state_trained", lambda value: "yes", "state branch"),
        ("schema_version", lambda value: 2, "schema_version=1"),
        ("score_schema_version", lambda value: 2, "score_schema_version=1"),
    ],
)
def test_replay_rejects_incomplete_or_misaligned_full_gallery_evidence(
    tiny_ranker, field, mutate, message
):
    source = _rank(tiny_ranker, top_m=1)
    source[field] = mutate(source[field])
    with pytest.raises(ValueError, match=message):
        replay_rankings(source, coarse_beta=2.0)


def test_export_does_not_enable_an_untrained_state_branch(tiny_ranker):
    source = copy.deepcopy(_rank(tiny_ranker, state_trained=False, coarse_beta=0.0))
    assert source["state_scores"].shape == source["identity_scores"].shape
    assert not source["state_trained"]
    source["state_scores"].fill_(torch.nan)  # beta=0 must ignore the whole branch.
    assert torch.equal(replay_rankings(source)["rankings"], source["rankings"])
    with pytest.raises(ValueError, match="state branch is untrained"):
        replay_rankings(source, coarse_beta=0.4)


def test_beta_alone_can_admit_a_previously_unscored_shortlist_candidate(tiny_ranker):
    source = copy.deepcopy(_rank(tiny_ranker, top_m=1, coarse_beta=0.0))
    old = source["coarse_topm"][0, 0].item()
    query = source["query_indices"][0].item()
    new = next(i for i in range(len(source["gallery_ids"])) if i not in (old, query))
    source["identity_scores"][0].zero_()
    source["identity_scores"][0, old] = 1.0
    source["state_scores"][0].zero_()
    source["state_scores"][0, new] = 1.0
    original = replay_rankings(source)
    changed = replay_rankings(original, coarse_beta=4.0)
    assert original["coarse_topm"][0, 0] == old
    assert changed["coarse_topm"][0, 0] == new
    assert (
        changed["fine_raw_scores"][0, 0]
        == source["binding_scale"] * source["fafa_binding"][0, new]
        + source["coverage"][0, new] * source["context"][0, new]
    )
    assert torch.equal(changed["rankings"][:, 1:], changed["coarse_rankings"][:, 1:])


def test_nonfinite_state_eligibility_is_recomputed_for_each_beta(tiny_ranker):
    source = copy.deepcopy(_rank(tiny_ranker, top_m=20, coarse_beta=0.0))
    index = source["coarse_topm"][0, 0].item()
    source["state_scores"][0, index] = torch.nan
    zero = replay_rankings(source, coarse_beta=0.0)
    positive = replay_rankings(source, coarse_beta=0.4)
    assert torch.isfinite(zero["coarse_topm_scores"][0]).all()
    position = (positive["coarse_topm"][0] == index).nonzero().item()
    assert torch.isneginf(positive["coarse_topm_scores"][0, position])
    assert torch.isneginf(positive["fused_topm_scores"][0, position])


def test_self_never_enters_coarse_statistics_or_fine_ranking(tiny_ranker):
    source = copy.deepcopy(_rank(tiny_ranker, top_m=20))
    expected = replay_rankings(source)
    for i, query in enumerate(source["query_indices"].tolist()):
        source["identity_scores"][i, query] = 3e38
        source["state_scores"][i, query] = -3e38
    changed = replay_rankings(source)
    assert torch.equal(expected["rankings"], changed["rankings"])
    torch.testing.assert_close(
        expected["coarse_topm_scores"], changed["coarse_topm_scores"], rtol=0, atol=0
    )
    for i, query in enumerate(source["query_indices"]):
        assert not (changed["coarse_rankings"][i] == query).any()
        assert not (changed["coarse_topm"][i] == query).any()


def test_replay_cache_reuses_statistics_and_matches_uncached_results(
    tiny_ranker, monkeypatch
):
    source = _rank(tiny_ranker, top_m=1)
    configs = [
        dict(coarse_beta=b, top_m=m, context_weight=c, fine_coarse_weight=f)
        for b, m, c, f in product([0.0, 0.4], [1, 4], [0.0, 0.13], [0.0, 0.4, 0.7])
    ]
    expected = [replay_rankings(source, **cfg) for cfg in configs]
    counts = {
        "combine_scores": 0,
        "_score_statistics": 0,
        "fine_score_components": 0,
        "_zscore_1d": 0,
    }
    for name in counts:
        original = getattr(replay, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(replay, name, counted)
    cache = {}
    for cfg, reference in zip(configs, expected, strict=True):
        actual = replay._replay_rankings(
            source, {**source["score_config"], **cfg}, cache
        )
        for key in (
            "coarse_rankings",
            "rankings",
            "fine_rankings",
            "fafa_rankings",
            "fine_scores",
            "fused_topm_scores",
            "context_mean",
            "context_std",
        ):
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
    q = len(source["sample_ids"])
    assert counts == {
        "combine_scores": q * 2,
        "_score_statistics": q * 2 * 2,
        "fine_score_components": q * 2 * 2 * 2,
        "_zscore_1d": q * 2 * 2 * 3,
    }


def test_full_export_save_load_cli_uses_embedded_metadata_only(tiny_ranker, tmp_path):
    source = _rank(tiny_ranker, top_m=1, coarse_beta=0.0)
    source["provenance"] = {
        "split": "val",
        "retrieval": tiny_ranker[4],
        "config": {
            "retrieval": tiny_ranker[4],
            "selected_checkpoint_sha256": "old",
            "selected_validation_sha256": "old",
            "selected_retrieval": {"coarse_beta": 0.0},
        },
    }
    save_results(tmp_path / "source", source, {"unrelated": "must not be read"})
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
    assert metadata == saved["provenance"]
    assert metadata["retrieval"] == metadata["config"]["retrieval"]
    assert metadata["retrieval"]["coarse_beta"] == 2.0
    assert metadata["retrieval"]["top_m"] == 4
    assert "selected_retrieval" not in metadata["config"]
    assert metadata["replay"]["source_sha256"] == digest == sha256_file(path)
    assert "unrelated" not in metadata


def test_standalone_validation_defaults_to_full_scores_and_matches_official_metrics(
    saved_pipeline,
):
    cfg, data, _ = saved_pipeline
    cfg["retrieval"].update(top_m=1, coarse_beta=0.0)
    exported = retrieve.retrieve(cfg)
    directory = Path(cfg["output"]["dir"]) / "val"
    saved = torch.load(directory / "rankings.pt", weights_only=True)
    assert torch.equal(saved["rankings"], exported["rankings"])
    assert saved["identity_scores"].shape == saved["state_scores"].shape == (1, 3)
    metadata = json.loads((directory / "run.json").read_text())
    assert saved["provenance"] == metadata
    assert saved["provenance"]["checkpoint_sha256"] == sha256_file(cfg["checkpoint"])
    replayed = replay_rankings(saved, coarse_beta=1.5, top_m=2, fine_coarse_weight=0.1)
    direct_cfg = copy.deepcopy(cfg)
    direct_cfg["output"]["dir"] = str(directory.parent / "direct")
    direct_cfg["retrieval"].update(coarse_beta=1.5, top_m=2, fine_coarse_weight=0.1)
    direct = retrieve.retrieve(direct_cfg)
    assert torch.equal(replayed["rankings"], direct["rankings"])
    assert evaluate_retrieval_output(
        data, [data.samples[1]], replayed, [1, 2], split="val"
    ) == evaluate_retrieval_output(data, [data.samples[1]], direct, [1, 2], split="val")


def test_offline_calibration_selects_and_freezes_without_checkpoint_or_caches(
    saved_pipeline, tmp_path, monkeypatch
):
    from rcr.evaluation.report import load_run

    cfg, data, _ = saved_pipeline
    cfg["retrieval"].update(top_m=1, coarse_beta=0.0)
    source = retrieve.retrieve(cfg)
    source_path = tmp_path / "scores-only.pt"
    torch.save(source, source_path)
    digest = sha256_file(source_path)
    checkpoint = Path(cfg["checkpoint"])
    moved = tmp_path / "hidden.pt"
    checkpoint.rename(moved)
    root = tmp_path / "offline-calibration"
    original_load = torch.load
    opened = []

    def only_scores(path, *args, **kwargs):
        assert Path(path) == source_path
        opened.append(Path(path))
        return original_load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "offline calibration must not load a model or cache, or run inference"
        )

    with monkeypatch.context() as guard:
        guard.setattr(torch, "load", only_scores)
        guard.setattr(retrieve, "load_text_modules", forbidden)
        guard.setattr(retrieve, "GalleryCache", forbidden)
        guard.setattr(RCRModel, "from_config", forbidden)
        guard.setattr(RCRModel, "score_target", forbidden)
        rows = calibration.calibrate(_settings(source_path, root))
    assert opened == [source_path]
    assert len(rows) == 17  # Source configuration plus sixteen grid candidates.
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    expected = torch.load(root / "selected/val/rankings.pt", weights_only=True)
    best = max(rows, key=lambda row: row["checkpoint_score"])
    for key in ("coarse_beta", "fine_coarse_weight", "context_weight", "top_m"):
        assert selected["retrieval"][key] == best[key] == expected["score_config"][key]
        assert selected["selected_retrieval"][key] == best[key]
    assert selected["candidate_ks"] == cfg["candidate_ks"] == [1, 2]
    assert all("candidate_recall_2" in row for row in rows)
    assert (
        selected["selected_checkpoint_sha256"]
        == source["provenance"]["checkpoint_sha256"]
    )
    assert (
        selected["selected_validation_sha256"]
        == source["provenance"]["validation_sha256"]
    )
    assert sha256_file(source_path) == digest
    assert (
        load_run("proposed", root / "selected/val", "val")["metrics"]["overall"][
            "full_map"
        ]
        == best["full_map"]
    )
    assert not list(root.glob("grid_*"))
    assert {"summary.csv", "selection.json", "selected.yaml"} <= {
        path.name for path in root.iterdir()
    }
    moved.rename(checkpoint)
    selected["output"]["dir"] = str(tmp_path / "direct-selected")
    assert torch.equal(retrieve.retrieve(selected)["rankings"], expected["rankings"])


@pytest.mark.parametrize(
    "changed",
    [
        "query_subset",
        "validation_sha256",
        "checkpoint_sha256",
        "gallery_ids",
        "selected_checkpoint_sha256",
        "split",
    ],
)
def test_calibration_rejects_invalid_provenance_before_writes(
    saved_pipeline, tmp_path, changed
):
    cfg, _, _ = saved_pipeline
    source = retrieve.retrieve(cfg)
    if changed == "gallery_ids":
        source["gallery_ids"].reverse()
    elif changed == "selected_checkpoint_sha256":
        source["provenance"]["config"][changed] = "f" * 64
    else:
        source["provenance"][changed] = True if changed == "query_subset" else "changed"
    path = tmp_path / "source.pt"
    torch.save(source, path)
    root = tmp_path / "rejected"
    with pytest.raises(
        ValueError,
        match="complete validation|changed since retrieval|checkpoint fingerprint",
    ):
        calibration.calibrate(_settings(path, root))
    assert not root.exists()


def test_selected_test_only_fine_scores_topm_and_checks_checkpoint(
    saved_pipeline, tmp_path, monkeypatch
):
    cfg, data, _ = saved_pipeline
    cfg["retrieval"]["top_m"] = 1
    retrieve.retrieve(cfg)
    root = tmp_path / "calibration"
    settings = _settings(Path(cfg["output"]["dir"]) / "val/rankings.pt", root)
    settings["top_m"] = [1]
    calibration.calibrate(settings)
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    selected["split"] = "test"
    # Reuse the tiny cache's other images as a separate test gallery.
    sample = _sample("test", "q", "a")
    data.samples[0] = sample
    data.samples_by_id = {s["sample_id"]: s for s in data.samples}
    data.splits.update(train=[], test=["test"])
    for image in ("q", "a", "n"):
        data.images_by_id[image]["path"] = f"test/{image}.jpg"
    score = RCRModel.score_target
    pairs = []

    def counted(self, query, scene, *args, **kwargs):
        pairs.extend(scene)
        return score(self, query, scene, *args, **kwargs)

    monkeypatch.setattr(RCRModel, "score_target", counted)
    output = retrieve.retrieve(selected)
    assert len(pairs) == 1
    assert output["rankings"].shape == (1, 2)
    assert output["fafa_binding"].shape == (1, 1)
    assert output["score_config"]["score_alignment"] == "coarse_topm"
    checkpoint = torch.load(selected["checkpoint"], weights_only=True)
    checkpoint["epoch"] += 1
    torch.save(checkpoint, selected["checkpoint"])
    with pytest.raises(ValueError, match="checkpoint changed since val selection"):
        retrieve.retrieve(selected)


@pytest.mark.parametrize("split", ["val", "test"])
def test_run_uses_configured_split_and_checkpoint(tmp_path, monkeypatch, split):
    from rcr.proposed import runner

    cfg = yaml.safe_load(Path("configs/proposed.yaml").read_text())
    cfg["output"]["dir"] = str(tmp_path / "run")
    cfg["split"] = split
    cfg["checkpoint"] = "frozen/best.pt"
    cfg["selected_checkpoint_sha256"] = "frozen-checkpoint"
    original, calls = copy.deepcopy(cfg), []

    def stage(command, **kwargs):
        calls.append(command)
        if command[2] == "evaluate":
            directory = Path(cfg["output"]["dir"]) / split
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "metrics.json").write_text(json.dumps({"overall": {}}))

    monkeypatch.setattr(runner.subprocess, "run", stage)
    rows = runner.run_experiment(cfg, Path("tools/run.py"))
    assert [cmd[2] for cmd in calls] == ["retrieve", "evaluate"]
    assert all(
        yaml.safe_load(Path(command[4]).read_text()) == original for command in calls
    )
    assert cfg == original
    assert rows == [
        {"method": "proposed", "split": split, "checkpoint": "frozen/best.pt"}
    ]


def test_misspelled_saved_score_setting_cannot_start_model_calibration():
    with pytest.raises(ValueError, match="unknown calibration settings: ranking"):
        calibration.calibrate({"ranking": "scores.pt"})


def test_candidate_recall_uses_full_coarse_order_independently_of_topm(saved_pipeline):
    cfg, data, _ = saved_pipeline
    cfg["retrieval"]["top_m"] = 1
    output = retrieve.retrieve(cfg)
    by_id = {image: i for i, image in enumerate(output["gallery_ids"])}
    output["coarse_rankings"] = torch.tensor([[by_id["vn"], by_id["va"]]])
    output["coarse_topm"] = output["coarse_rankings"][:, :1]
    metrics = evaluate_retrieval_output(
        data, [data.samples[1]], output, [1, 2], split="val"
    )
    assert metrics["overall"]["candidate_recall_1"] == 0
    assert metrics["overall"]["candidate_recall_2"] == 1


def test_calibration_cli_runs_with_no_model_imports_and_only_real_annotations(
    saved_pipeline, tmp_path
):
    import subprocess
    import sys

    from rcr.common.io import write_json, write_jsonl

    cfg, data, _ = saved_pipeline
    final = tmp_path / "annotations"
    cfg["data"]["final_dir"] = str(final)
    cfg["data"]["image_root"] = str(tmp_path / "images-not-present")
    data.gt_head_boxes_by_image = {
        image: [{"image_id": image, **box} for box in boxes]
        for image, boxes in data.gt_head_boxes_by_image.items()
    }
    write_jsonl(final / "samples.jsonl", data.samples)
    write_jsonl(final / "images.jsonl", data.images_by_id.values())
    write_jsonl(
        final / "gallery.jsonl", [{"image_id": image} for image in data.gallery_ids]
    )
    write_jsonl(
        final / "head_boxes.jsonl",
        [
            {"image_id": image, **box}
            for image, boxes in data.gt_head_boxes_by_image.items()
            for box in boxes
        ],
    )
    write_json(final / "manifest.json", {})
    (final / "splits").mkdir()
    for split, ids in data.splits.items():
        (final / "splits" / f"{split}.txt").write_text("\n".join(ids))
    source = retrieve.retrieve(cfg)
    scores = tmp_path / "standalone-scores.pt"
    torch.save(source, scores)
    Path(cfg["checkpoint"]).unlink()
    settings = _settings(scores, tmp_path / "calibrated")
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(settings))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('transformers', 'rcr.proposed.nn', 'rcr.proposed.cache',
                                'rcr.proposed.retrieve')):
            raise AssertionError('offline calibration imported ' + fullname)
sys.meta_path.insert(0, NoModels())
from tools.run import main
main(['calibrate', '--config', sys.argv[1]])
""",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = yaml.safe_load(
        (Path(settings["output_dir"]) / "selected.yaml").read_text()
    )
    assert selected["candidate_ks"] == cfg["candidate_ks"]
    assert (
        selected["selected_checkpoint_sha256"]
        == source["provenance"]["checkpoint_sha256"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("top_m", [0]),
        ("top_m", [1.5]),
        ("top_m", [True]),
        ("coarse_beta", [-0.1]),
        ("fine_coarse_weight", [float("nan")]),
        ("context_weight", []),
    ],
)
def test_invalid_grids_fail_before_opening_rankings(tmp_path, key, value):
    settings = _settings(tmp_path / "not-present.pt", tmp_path / "output")
    settings[key] = value
    with pytest.raises(ValueError, match=key):
        calibration.calibrate(settings)
    assert not (tmp_path / "output").exists()
