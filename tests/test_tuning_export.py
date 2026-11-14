"""CPU workflow tests for Top-M score export, offline replay and selection."""

from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.io import sha256_file
from rcr.proposed import calibration, retrieve
from rcr.proposed.nn.model import RCRModel
from rcr.proposed.ranking import SCORE_SCHEMA_VERSION
from rcr.proposed.replay import replay_rankings
from rcr.proposed.scores import POOLING_MODES
from tests.helpers import _rank, _sample


def _settings(paths, root):
    return {"rankings": [str(path) for path in paths], "output_dir": str(root)}


def test_retrieval_scores_each_shortlist_pair_once_and_stores_topm_evidence(
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
        == 3
    )
    assert "score_alignment" not in output["score_config"]
    assert {"identity_scores", "state_scores", "tuning_scores"}.isdisjoint(output)
    expected = {
        (cache.by_id[sample["query_image_id"]], int(output["coarse_topm"][n, 0]))
        for n, sample in enumerate(samples)
    }
    assert len(pairs) == len(set(pairs)) == len(samples)
    assert set(pairs) == expected
    for key in (
        "fafa_binding",
        "context",
        "coverage",
        *(f"binding_{mode}" for mode in POOLING_MODES),
    ):
        assert output[key].shape == (len(samples), 1)
        assert output[key].dtype == torch.float32
        assert output[key].device.type == "cpu"
    binding, context, coverage = (
        output[key] for key in ("fafa_binding", "context", "coverage")
    )
    torch.testing.assert_close(
        output["fine_scores"],
        output["binding_scale"] * binding
        + output["score_config"]["context_weight"]
        * (coverage * context.tanh() - (1 - coverage)),
        rtol=0,
        atol=0,
    )
    pooling = output["score_config"]["pooling"]
    assert torch.equal(output["fafa_binding"], output[f"binding_{pooling}"])
    assert {
        "case_type",
        "identity_ids",
        "positive_image_ids",
        "target_image_id",
    }.isdisjoint(output)
    assert torch.equal(replay_rankings(output)["rankings"], output["rankings"])


@pytest.mark.parametrize("pooling", POOLING_MODES)
def test_replay_matches_direct_retrieval_for_fine_score_settings(tiny_ranker, pooling):
    source = _rank(tiny_ranker)
    parameters = dict(pooling=pooling, context_weight=0.13)
    replayed = replay_rankings(source, **parameters)
    direct = _rank(tiny_ranker, **parameters)
    for key in (
        "coarse_rankings",
        "coarse_topm",
        "fine_rankings",
        "fafa_rankings",
        "rankings",
    ):
        assert torch.equal(replayed[key], direct[key]), key
    for key in ("coarse_topm_scores", "fine_scores", "fafa_binding", "context"):
        torch.testing.assert_close(replayed[key], direct[key], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        replayed["coverage"], direct["coverage"], atol=2e-6, rtol=2e-5
    )
    assert replayed["score_config"]["pooling"] == pooling
    assert replayed["provenance"]["fine_score_ablation"]
    assert "fine_score_ablation" not in replay_rankings(source)["provenance"]


def test_replay_rejects_settings_that_change_the_shortlist(tiny_ranker):
    source = _rank(tiny_ranker)
    recorded = source["score_config"]
    for parameters in (
        {"coarse_beta": recorded["coarse_beta"] + 1.0},
        {"top_m": recorded["top_m"] - 1},
    ):
        with pytest.raises(ValueError, match="changes the Top-M shortlist"):
            replay_rankings(source, **parameters)
    unchanged = replay_rankings(
        source, coarse_beta=recorded["coarse_beta"], top_m=recorded["top_m"]
    )
    assert torch.equal(unchanged["rankings"], source["rankings"])


def test_offline_calibration_selects_and_freezes_without_checkpoint_or_caches(
    saved_pipeline, tmp_path, monkeypatch
):
    from rcr.evaluation.report import load_run

    cfg, data, _ = saved_pipeline
    cfg["retrieval"]["top_m"] = 1
    paths, sources = [], []
    for name, beta in (("zero", 0.0), ("tuned", 1.5)):
        cfg["retrieval"]["coarse_beta"] = beta
        sources.append(retrieve.retrieve(cfg))
        paths.append(tmp_path / f"{name}-scores.pt")
        torch.save(sources[-1], paths[-1])
    digests = [sha256_file(path) for path in paths]
    checkpoint = Path(cfg["checkpoint"])
    moved = tmp_path / "hidden.pt"
    checkpoint.rename(moved)
    root = tmp_path / "offline-calibration"
    original_load = torch.load
    opened = []

    def only_scores(path, *args, **kwargs):
        assert Path(path) in paths
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
        rows = calibration.calibrate(_settings(paths, root))
    assert opened == paths
    assert [row["coarse_beta"] for row in rows] == [0.0, 1.5]
    selected = yaml.safe_load((root / "selected.yaml").read_text())
    expected = torch.load(root / "selected/val/rankings.pt", weights_only=True)
    best = max(rows, key=lambda row: row["checkpoint_score"])
    for key in ("coarse_beta", "context_weight", "top_m"):
        assert selected["retrieval"][key] == best[key] == expected["score_config"][key]
        assert selected["selected_retrieval"][key] == best[key]
    assert selected["candidate_ks"] == cfg["candidate_ks"] == [1, 2]
    assert all("candidate_recall_2" in row for row in rows)
    source = sources[best["index"]]
    assert torch.equal(expected["rankings"], source["rankings"])
    assert (
        selected["selected_checkpoint_sha256"]
        == source["provenance"]["checkpoint_sha256"]
    )
    assert (
        selected["selected_validation_sha256"]
        == source["provenance"]["validation_sha256"]
    )
    assert [sha256_file(path) for path in paths] == digests
    assert (
        load_run("proposed", root / "selected/val", "val")["metrics"]["overall"][
            "full_map"
        ]
        == best["full_map"]
    )
    assert {"summary.csv", "selection.json", "selected.yaml"} <= {
        path.name for path in root.iterdir()
    }
    moved.rename(checkpoint)
    selected["output"]["dir"] = str(tmp_path / "direct-selected")
    assert torch.equal(retrieve.retrieve(selected)["rankings"], expected["rankings"])


def test_selected_test_only_fine_scores_topm_and_checks_checkpoint(
    saved_pipeline, tmp_path, monkeypatch
):
    cfg, data, _ = saved_pipeline
    cfg["retrieval"]["top_m"] = 1
    retrieve.retrieve(cfg)
    root = tmp_path / "calibration"
    calibration.calibrate(
        _settings([Path(cfg["output"]["dir"]) / "val/rankings.pt"], root)
    )
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
    checkpoint = torch.load(selected["checkpoint"], weights_only=True)
    checkpoint["epoch"] += 1
    torch.save(checkpoint, selected["checkpoint"])
    with pytest.raises(ValueError, match="checkpoint changed since val selection"):
        retrieve.retrieve(selected)


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
    settings = _settings([scores], tmp_path / "calibrated")
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
