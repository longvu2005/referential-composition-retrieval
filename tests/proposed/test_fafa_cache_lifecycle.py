"""User workflows: independent stages, text edits, order changes and recovery."""

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from rcr.common.data import load_rcr_data
from rcr.proposed.cache import build as builder
from rcr.proposed.cache import fafa as extraction
from rcr.proposed.cache.native import NativeQueryCache, query_entries
from rcr.proposed.cache.store import GalleryCache
from tests.proposed.test_runner import experiment as experiment
from tests.proposed.test_runner import local_stages as local_stages
from tests.proposed.test_split_cache import split_cache as split_cache
from tools import run as cli


def _text_limit(cfg, length):
    path = Path(cfg["person_encoder"]["fafa_config"])
    native = yaml.safe_load(path.read_text())
    native["checkpoint"]["max_txt_len"] = length
    path.write_text(yaml.safe_dump(native))


def _files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _rows(cfg):
    return load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"]).samples


def _save_rows(cfg, rows):
    path = Path(cfg["data"]["final_dir"]) / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _gallery(cfg):
    return GalleryCache(
        cfg["data"]["fafa_image_cache"],
        scene_root=cfg["data"]["dino_cache"],
        query_root=cfg["data"]["fafa_query_cache"],
    )


def _two_subjects(cfg):
    rows = _rows(cfg)
    rows[0]["subjects"] = [{"subject_id": 1}, {"subject_id": 2}]
    rows[0]["final_change"] = "Subject 1 wears red and Subject 2 wears blue"
    _save_rows(cfg, rows)
    return rows


def _record(model, monkeypatch):
    calls = []
    original = model.extract_features

    def extract(samples, mode):
        calls.append(mode)
        result = original(samples, mode)
        if mode == "multimodal":
            # Depend on actual input text so stale/misordered features are visible.
            result.multimodal_embeds = torch.tensor(
                [
                    [sum(text.encode()) / 1000, len(text) / 100, 1.0]
                    for text in samples["text_input"]
                ]
            )
        return result

    monkeypatch.setattr(model, "extract_features", extract)
    return calls


def test_independent_stages_keep_readonly_sources_untouched(split_cache, monkeypatch):
    cfg, path, source, model, _ = split_cache
    calls = _record(model, monkeypatch)
    before = _files(source)
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "fafa-image"])
    image_root = Path(cfg["data"]["fafa_image_cache"])
    query_root = Path(cfg["data"]["fafa_query_cache"])
    assert not query_root.exists()
    assert calls == ["image"] * len(
        GalleryCache(image_root, scene_root=source).image_ids
    )
    index = torch.load(image_root / "index.pt", weights_only=True)
    assert "queries" not in index["native"] and "queries" not in index["signature"]
    image_files = _files(image_root)
    calls.clear()
    alias = image_root.with_name("readonly-image-input")
    alias.symlink_to(image_root, target_is_directory=True)
    cfg["data"]["fafa_image_cache"] = str(alias)
    path.write_text(yaml.safe_dump(cfg))
    save = extraction.save_torch

    def query_writes_only(value, target):
        assert query_root.resolve() in target.resolve().parents
        return save(value, target)

    monkeypatch.setattr(extraction, "save_torch", query_writes_only)
    cli.main(["build-cache", "--config", str(path), "--cache-stage", "fafa-query"])
    assert calls == ["multimodal"] * len(_rows(cfg))
    assert _files(image_root) == image_files and _files(source) == before
    _gallery(cfg).native_queries(_rows(cfg), 1)
    # No raw images, native model or worker is needed for completed cache reuse.
    monkeypatch.setattr(extraction, "load_fafa", lambda *a: pytest.fail("loaded FAFA"))
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *a, **k: pytest.fail("worker")
    )
    cfg["data"]["image_root"] = "missing-images"
    builder.prepare_cache(cfg)


def test_edit_one_subject_only_extracts_one_new_condition(split_cache, monkeypatch):
    cfg, _, source, model, _ = split_cache
    rows = _two_subjects(cfg)
    calls = _record(model, monkeypatch)
    extraction.finish_fafa_cache(cfg)
    gallery = _gallery(cfg)
    old = gallery.native_queries(rows[:1], 1)
    image_root = Path(cfg["data"]["fafa_image_cache"])
    query_root = Path(cfg["data"]["fafa_query_cache"])
    saved_images, saved_source = _files(image_root), _files(source)
    saved_queries = _files(query_root / "features")
    image_id, query_id = gallery.cache_id, gallery.native.query_cache.cache_id
    calls.clear()
    rows[0]["final_change"] = "Subject 1 wears green and Subject 2 wears blue"
    _save_rows(cfg, rows)
    with pytest.raises(ValueError, match="stale"):
        gallery.validate_samples(rows)
    builder.prepare_cache(cfg, stage="fafa-query")
    assert calls == ["multimodal"]
    updated = _gallery(cfg)
    new = updated.native_queries(rows[:1], 1)
    assert not torch.equal(old[:, 0], new[:, 0])
    torch.testing.assert_close(old[:, 1], new[:, 1], rtol=0, atol=0)
    assert updated.cache_id == image_id
    assert updated.native.query_cache.cache_id == query_id
    assert _files(image_root) == saved_images and _files(source) == saved_source
    after = _files(query_root / "features")
    assert len(after) == len(saved_queries) + 1
    assert all(after[key] == value for key, value in saved_queries.items())


def test_sample_and_subject_reordering_reuses_exact_tensors(split_cache, monkeypatch):
    cfg, _, _, model, _ = split_cache
    rows = _two_subjects(cfg)
    _record(model, monkeypatch)
    extraction.finish_fafa_cache(cfg)
    old = _gallery(cfg).native_queries(rows, 1)
    rows[0]["subjects"].reverse()
    rows.reverse()
    _save_rows(cfg, rows)
    query_root = Path(cfg["data"]["fafa_query_cache"])
    before = _files(query_root / "features")
    monkeypatch.setattr(extraction, "load_fafa", lambda *a: pytest.fail("loaded FAFA"))
    builder.prepare_cache(cfg, stage="fafa-query")
    new = _gallery(cfg).native_queries(rows, 1)
    torch.testing.assert_close(old[0, :2].flip(0), new[-1, :2], rtol=0, atol=0)
    torch.testing.assert_close(old[1:, :1].flip(0), new[:-1, :1], rtol=0, atol=0)
    assert _files(query_root / "features") == before


def test_query_stage_can_precede_image_stage(split_cache, monkeypatch):
    cfg, _, _, model, _ = split_cache
    calls = _record(model, monkeypatch)
    builder.prepare_cache(cfg, stage="fafa-query")
    assert calls and set(calls) == {"multimodal"}
    assert not Path(cfg["data"]["fafa_image_cache"]).exists()
    before = _files(Path(cfg["data"]["fafa_query_cache"]))
    calls.clear()
    builder.prepare_cache(cfg, stage="fafa-image")
    assert calls and set(calls) == {"image"}
    assert _files(Path(cfg["data"]["fafa_query_cache"])) == before
    _gallery(cfg).validate_samples(_rows(cfg))


def test_query_resume_accepts_changed_instruction_and_keeps_image_cache(
    split_cache, monkeypatch
):
    cfg, _, source, model, _ = split_cache
    extraction.finish_fafa_cache(cfg, stage="fafa-image")
    image_root = Path(cfg["data"]["fafa_image_cache"])
    image_files = _files(image_root)
    source_files = _files(source)
    original = model.extract_features
    count = 0

    def interrupt(samples, mode):
        nonlocal count
        assert mode == "multimodal"
        count += 1
        if count == 2:
            raise RuntimeError("interrupted")
        return original(samples, mode)

    monkeypatch.setattr(model, "extract_features", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        extraction.finish_fafa_cache(cfg, stage="fafa-query")
    query_root = Path(cfg["data"]["fafa_query_cache"])
    saved = _files(query_root / "features")
    assert len(saved) == 1
    # Image-only consumers still work while query extraction is incomplete.
    GalleryCache(image_root, scene_root=source).native_targets(torch.tensor([0]), 1)
    with pytest.raises(RuntimeError, match="incomplete"):
        _gallery(cfg)
    rows = _rows(cfg)
    rows[-1]["final_change"] += " and wears green"
    _save_rows(cfg, rows)
    monkeypatch.setattr(model, "extract_features", original)
    calls = _record(model, monkeypatch)
    builder.prepare_cache(cfg, stage="fafa-query")
    assert calls == ["multimodal"] * 2
    assert _files(image_root) == image_files and _files(source) == source_files
    after = _files(query_root / "features")
    assert all(after[key] == value for key, value in saved.items())
    _gallery(cfg).validate_samples(rows)


def test_image_resume_ignores_query_changes(split_cache, monkeypatch):
    cfg, _, _, model, _ = split_cache
    original = model.extract_features
    calls = 0

    def interrupt(samples, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(samples, mode)

    monkeypatch.setattr(model, "extract_features", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        extraction.finish_fafa_cache(cfg, stage="fafa-image")
    image_root = Path(cfg["data"]["fafa_image_cache"])
    saved = (image_root / "targets/0.pt").read_bytes()
    rows = _rows(cfg)
    rows[0]["final_change"] += " and wears green"
    _save_rows(cfg, rows)
    monkeypatch.setattr(model, "extract_features", original)
    remaining = _record(model, monkeypatch)
    builder.prepare_cache(cfg, stage="fafa-image")
    assert remaining == ["image"] * 8
    assert (image_root / "targets/0.pt").read_bytes() == saved
    assert not Path(cfg["data"]["fafa_query_cache"]).exists()


def test_query_manifest_rejects_another_encoder_or_crop_family(split_cache):
    cfg, _, _, _, _ = split_cache
    extraction.finish_fafa_cache(cfg)
    gallery = _gallery(cfg)
    signature = copy.deepcopy(gallery.feature_signature)
    signature["source_cache_id"] = "other-crops"
    with pytest.raises(ValueError, match="encoder/crops differ"):
        NativeQueryCache(cfg["data"]["fafa_query_cache"], signature, gallery.native.dim)
    rows = _rows(cfg)
    unlabeled = copy.deepcopy(rows)
    for row in unlabeled:
        row["case_type"] = "RELATIONAL"
        row["positive_image_ids"] = ["unseen-ground-truth"]
        for subject in row["subjects"]:
            subject["identity_ids"] = ["different-identity-label"]
    assert query_entries(unlabeled, signature) == query_entries(rows, signature)


@pytest.mark.parametrize("alias", ["same", "nested", "symlink"])
def test_cache_aliases_fail_before_any_writes(split_cache, alias):
    cfg, _, source, _, _ = split_cache
    before = _files(source)
    if alias == "same":
        cfg["data"]["fafa_query_cache"] = cfg["data"]["fafa_image_cache"]
    elif alias == "nested":
        cfg["data"]["fafa_query_cache"] = str(source / "queries")
    else:
        link = source.with_name("source-alias")
        link.symlink_to(source, target_is_directory=True)
        cfg["data"]["fafa_image_cache"] = str(link)
    with pytest.raises(ValueError, match="separate directories"):
        builder.prepare_cache(cfg)
    assert _files(source) == before


def test_completed_unified_images_are_reused_without_writes(split_cache, monkeypatch):
    cfg, _, source, _, _ = split_cache
    extraction.finish_fafa_cache(cfg)
    image_root = Path(cfg["data"]["fafa_image_cache"])
    image_index = torch.load(image_root / "index.pt", weights_only=True)
    cache = GalleryCache(image_root, scene_root=source)
    image_index["persons"], _ = cache.load_persons(torch.arange(len(cache.image_ids)))
    image_index.pop("person_offsets")
    image_index["format_version"] = 4
    image_index.pop("signature")
    # Legacy embedded query metadata must not constrain the separate query cache.
    image_index["native"]["queries"] = {"obsolete-sample": {"index": 0, "key": "old"}}
    torch.save(image_index, image_root / "index.pt")
    saved_images = _files(image_root)
    cfg["data"]["fafa_query_cache"] += "-new"
    original = extraction.load_fafa

    def load(*args):
        model, *rest = original(*args)
        extract = model.extract_features

        def query_only(samples, mode):
            assert mode == "multimodal"
            return extract(samples, mode)

        monkeypatch.setattr(model, "extract_features", query_only)
        return model, *rest

    monkeypatch.setattr(extraction, "load_fafa", load)
    builder.prepare_cache(cfg, stage="fafa")
    assert _files(image_root) == saved_images
    gallery = _gallery(cfg)
    assert gallery.cache_id == image_index["cache_id"]
    assert gallery.feature_signature["source_cache_id"] == GalleryCache(source).cache_id
    gallery.native_queries(_rows(cfg), 1)


def test_failed_query_disk_check_keeps_previous_index_and_features(
    split_cache, monkeypatch
):
    cfg, _, _, model, _ = split_cache
    extraction.finish_fafa_cache(cfg)
    root = Path(cfg["data"]["fafa_query_cache"])
    before_index = (root / "index.pt").read_bytes()
    saved = _files(root / "features")
    rows = _rows(cfg)
    rows[0]["final_change"] += " different shoes"
    _save_rows(cfg, rows)
    guard = extraction.require_disk_space

    def no_space(directory, size):
        raise OSError("no space")

    calls = _record(model, monkeypatch)
    monkeypatch.setattr(extraction, "require_disk_space", no_space)
    with pytest.raises(OSError, match="no space"):
        builder.prepare_cache(cfg, stage="fafa-query")
    assert not calls
    assert (root / "index.pt").read_bytes() == before_index
    assert _files(root / "features") == saved
    with pytest.raises(RuntimeError, match="incomplete"):
        _gallery(cfg)
    monkeypatch.setattr(extraction, "require_disk_space", guard)
    builder.prepare_cache(cfg, stage="fafa-query")
    assert calls == ["multimodal"]
    _gallery(cfg).validate_samples(rows)


def test_equivalent_rebuilt_query_cache_works_with_saved_checkpoint(split_cache):
    cfg, path, _, _, _ = split_cache
    cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    old = _gallery(cfg)
    cfg["data"]["fafa_query_cache"] += "-rebuilt"
    path.write_text(yaml.safe_dump(cfg))
    builder.prepare_cache(cfg, stage="fafa-query")
    new = _gallery(cfg)
    assert new.native.query_cache.cache_id != old.native.query_cache.cache_id
    assert new.native.metadata == old.native.metadata
    torch.testing.assert_close(
        new.native_queries(_rows(cfg), 1),
        old.native_queries(_rows(cfg), 1),
        rtol=0,
        atol=0,
    )
    cli.main(["retrieve", "--config", str(path), "--splits", "test"])


@pytest.mark.parametrize("stage", ["fafa", "fafa-image", "fafa-query"])
def test_stage_reaches_isolated_worker_entrypoint(experiment, monkeypatch, stage):
    from tools import cache_fafa

    cfg, path = experiment
    received = []
    monkeypatch.setattr(
        extraction, "finish_fafa_cache", lambda cfg, **kw: received.append((cfg, kw))
    )
    cache_fafa.main(["--config", str(path), "--stage", stage])
    assert received == [(cfg, {"stage": stage})]


def test_query_provenance_mismatch_fails_before_extracting_images(
    split_cache, monkeypatch
):
    cfg, _, _, _, _ = split_cache
    extraction.finish_fafa_cache(cfg, stage="fafa-query")
    index = Path(cfg["data"]["fafa_query_cache"]) / "index.pt"
    item = torch.load(index, weights_only=True)
    item["signature"]["source_cache_id"] = "other-crops"
    torch.save(item, index)
    monkeypatch.setattr(extraction, "load_fafa", lambda *a: pytest.fail("loaded FAFA"))
    with pytest.raises(ValueError, match="encoder/crops differ"):
        extraction.finish_fafa_cache(cfg)
    assert not Path(cfg["data"]["fafa_image_cache"]).exists()


def test_text_limit_upgrade_rebuilds_only_queries(split_cache, monkeypatch):
    cfg, _, source, model, _ = split_cache
    _text_limit(cfg, 32)
    extraction.finish_fafa_cache(cfg)
    old = _gallery(cfg)
    old_queries = {
        key for row in old.native.queries.values() for key in row["features"]
    }
    image_root = Path(cfg["data"]["fafa_image_cache"])
    query_root = Path(cfg["data"]["fafa_query_cache"])
    images, dino = _files(image_root), _files(source)
    calls = _record(model, monkeypatch)
    _text_limit(cfg, 128)
    monkeypatch.setattr(
        extraction, "_build_images", lambda *a: pytest.fail("rebuilt image features")
    )
    save = extraction.save_torch

    def queries_only(value, target):
        assert query_root in target.parents
        return save(value, target)

    monkeypatch.setattr(extraction, "save_torch", queries_only)
    builder.prepare_cache(cfg)
    new = _gallery(cfg)
    new_queries = {
        key for row in new.native.queries.values() for key in row["features"]
    }
    assert new.native.text_policy == {"max_txt_len": 128, "max_words": 128}
    assert not old_queries & new_queries
    assert new.native.query_cache.cache_id != old.native.query_cache.cache_id
    assert new.cache_id == old.cache_id
    assert calls == ["multimodal"] * len(_rows(cfg))
    assert _files(image_root) == images and _files(source) == dino
    monkeypatch.setattr(
        builder, "run_person_worker", lambda *a, **kw: pytest.fail("extracted twice")
    )
    builder.prepare_cache(cfg)


def test_training_rejects_stale_text_config_before_changing_output(split_cache):
    from rcr.proposed.train import train

    cfg, _, _, _, _ = split_cache
    _text_limit(cfg, 32)
    extraction.finish_fafa_cache(cfg)
    output = Path(cfg["output"]["dir"])
    output.mkdir(parents=True)
    (output / "best.pt").write_bytes(b"previous checkpoint")
    before = _files(output)
    _text_limit(cfg, 128)
    with pytest.raises(ValueError, match="query text policy differs from config"):
        train(cfg)
    assert _files(output) == before


def test_legacy_query_and_checkpoint_still_work_at_32_tokens(split_cache):
    cfg, path, _, _, _ = split_cache
    _text_limit(cfg, 32)
    extraction.finish_fafa_cache(cfg)
    query_index = Path(cfg["data"]["fafa_query_cache"]) / "index.pt"
    index = torch.load(query_index, weights_only=True)
    index.pop("text_policy")
    index.pop("text_stats")
    torch.save(index, query_index)
    assert _gallery(cfg).native.text_policy == {"max_txt_len": 32, "max_words": 50}
    cli.main(["train", "--config", str(path)])
    checkpoint = torch.load(cfg["checkpoint"], weights_only=True)
    checkpoint["native_cache"].pop("text_policy")
    torch.save(checkpoint, cfg["checkpoint"])
    cli.main(["retrieve", "--config", str(path), "--splits", "test"])


def test_checkpoint_rejects_query_features_with_different_text_limit(split_cache):
    cfg, path, _, _, _ = split_cache
    _text_limit(cfg, 32)
    cli.main(["run", "--config", str(path), "--build-cache", "--train"])
    saved = Path(cfg["checkpoint"]).read_bytes()
    _text_limit(cfg, 128)
    builder.prepare_cache(cfg, stage="fafa-query")
    with pytest.raises(ValueError, match="query text policy differs from training"):
        cli.main(["retrieve", "--config", str(path), "--splits", "test"])
    assert Path(cfg["checkpoint"]).read_bytes() == saved


def test_text_upgrade_resumes_without_mixing_limits(split_cache, monkeypatch):
    cfg, _, source, model, _ = split_cache
    _text_limit(cfg, 32)
    extraction.finish_fafa_cache(cfg)
    images = _files(Path(cfg["data"]["fafa_image_cache"]))
    dino = _files(source)
    _text_limit(cfg, 128)
    original, count = model.extract_features, 0

    def interrupted(samples, mode):
        nonlocal count
        assert mode == "multimodal"
        count += 1
        if count == 2:
            raise RuntimeError("interrupted text upgrade")
        return original(samples, mode)

    monkeypatch.setattr(model, "extract_features", interrupted)
    with pytest.raises(RuntimeError, match="interrupted text upgrade"):
        builder.prepare_cache(cfg, stage="fafa-query")
    query_root = Path(cfg["data"]["fafa_query_cache"])
    interrupted_files = _files(query_root)
    _text_limit(cfg, 64)
    with pytest.raises(ValueError, match="unfinished FAFA cache settings differ"):
        builder.prepare_cache(cfg, stage="fafa-query")
    assert _files(query_root) == interrupted_files
    _text_limit(cfg, 128)
    monkeypatch.setattr(model, "extract_features", original)
    remaining = _record(model, monkeypatch)
    builder.prepare_cache(cfg, stage="fafa-query")
    assert remaining == ["multimodal"] * (len(_rows(cfg)) - 1)
    _gallery(cfg).validate_samples(_rows(cfg))
    assert _files(Path(cfg["data"]["fafa_image_cache"])) == images
    assert _files(source) == dino


def test_query_build_reports_conditions_exceeding_token_limit(split_cache):
    cfg, _, _, _, _ = split_cache
    rows = _rows(cfg)
    rows[0]["final_change"] = "Subject 1 " + "wears red " * 80
    _save_rows(cfg, rows)
    extraction.finish_fafa_cache(cfg)
    index = torch.load(
        Path(cfg["data"]["fafa_query_cache"]) / "index.pt", weights_only=True
    )
    assert index["text_stats"]["conditions_checked"] == len(rows)
    assert index["text_stats"]["over_token_limit"] == 1
    assert index["text_stats"]["max_tokens"] > 128
