"""Exercise actual CLI entrypoints for append, pending work and rerun."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.rewrite import prepare_rewrite_inputs
from rcr.dataset.selection import select_samples
from scripts.data import build_final, prepare_handoffs, prepare_rewrite


def _source(query, target, identity, split):
    desc = "Identify Subject 1 as the person"
    change = "then retrieve target images where Subject 1 is smiling"
    box = dict(identity_id=identity, x=0.1, y=0.1, width=0.2, height=0.2)
    return dict(
        sample_id=f"{split.lower()}__{query}__{target}",
        split=split,
        annotator_email="test@example.org",
        query_image_id=query,
        query_image_path=f"{split.lower()}/{query}.jpg",
        query_boxes=[box],
        target_image_id=target,
        target_image_path=f"{split.lower()}/{target}.jpg",
        target_boxes=[dict(box)],
        case_type="INDIVIDUAL",
        subjects=[dict(subject_id=1, identity_ids=[identity])],
        final_desc=desc,
        final_change=change,
        final_instruction=f"{desc}; {change}.",
    )


def test_append_only_pipeline_with_optional_clip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = _source("1_1", "1_2", "7", "TRAIN")
    new = _source("1_3", "1_4", "8", "VAL")
    pair_data = {"pairs": [], "boxes": []}
    for row in (old, new):
        pair_data["pairs"].append(
            {
                "pair_id": row["sample_id"],
                "split": row["split"],
                "query_image_id": row["query_image_id"],
                "target_image_id": row["target_image_id"],
            }
        )
        for side in ("query", "target"):
            box = dict(row[f"{side}_boxes"][0])
            box["label"] = box.pop("identity_id")
            box["image_id"] = row[f"{side}_image_id"]
            pair_data["boxes"].append(box)
    lines = []
    for i in range(1, 9):
        identity = "7" if i in (1, 2, 5, 7) else "8"
        split = "train" if identity == "7" else "val"
        path = prepare_handoffs.IMAGE_ROOT / split / f"1_{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10)).save(path)
        lines.append(f"1 {i} 1 1 2 2 {identity} 1")
    prepare_handoffs.INDEX.parent.mkdir(parents=True)
    prepare_handoffs.INDEX.write_text("\n".join(lines) + "\n")
    build_final.PAIR_DATA.write_text(json.dumps(pair_data))

    def prepare(rows):
        write_jsonl(prepare_handoffs.SELECTED, select_samples(rows, pair_data))
        prepare_rewrite.main()
        monkeypatch.setattr(sys, "argv", ["prepare_handoffs.py", "review"])
        prepare_handoffs.main()

    def positives(clip=False):
        args = ["prepare_handoffs.py", "positives"]
        if clip:
            args.append("--clip-rerank")
        monkeypatch.setattr(sys, "argv", args)
        prepare_handoffs.main()

    prepare([old])
    old_review = prepare_rewrite_inputs([old])
    write_jsonl(prepare_handoffs.REVIEWED, old_review)
    positives()
    old_catalog = load_jsonl(prepare_handoffs.POSITIVE_INPUT)
    old_decision = {"sample_id": old["sample_id"], "positive_image_ids": ["1_2"]}
    write_jsonl(prepare_handoffs.POSITIVE_SETS, [old_decision])

    # Appending raw data does not invent completed review/positive decisions.
    prepare([old, new])
    assert len(load_jsonl(prepare_handoffs.REVIEW_INPUT)) == 2
    assert load_jsonl(prepare_handoffs.REVIEWED) == old_review
    calls = []

    class FakeRanker:
        def __init__(self):
            calls.append("loaded")

        def __call__(self, change, candidates):
            calls.append([image for image, _ in candidates])
            return [image for image, _ in reversed(candidates)]

    monkeypatch.setitem(
        sys.modules,
        "rcr.dataset.clip_rerank",
        SimpleNamespace(ClipChangeRanker=FakeRanker),
    )
    positives(clip=True)
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["build_final.py", "--allow-partial"])
    build_final.main()
    assert len(load_jsonl(build_final.PARTIAL_OUTPUT / "samples.jsonl")) == 1
    monkeypatch.setattr(sys, "argv", ["build_final.py"])
    with pytest.raises(ValueError, match="reviewed samples do not match"):
        build_final.main()

    write_jsonl(prepare_handoffs.REVIEWED, prepare_rewrite_inputs([old, new]))
    positives(clip=True)
    catalog = load_jsonl(prepare_handoffs.POSITIVE_INPUT)
    assert catalog[:1] == old_catalog
    assert calls == ["loaded", ["1_6", "1_8"]]
    assert [x["image_id"] for x in catalog[1]["candidates"]] == ["1_4", "1_8", "1_6"]
    assert load_jsonl(prepare_handoffs.POSITIVE_SETS) == [old_decision]
    before = prepare_handoffs.POSITIVE_INPUT.read_bytes()
    positives(clip=True)
    assert prepare_handoffs.POSITIVE_INPUT.read_bytes() == before
    assert calls == ["loaded", ["1_6", "1_8"]]

    write_jsonl(
        prepare_handoffs.POSITIVE_SETS,
        [
            old_decision,
            {
                "sample_id": new["sample_id"],
                "positive_image_ids": ["1_4", "1_8"],
            },
        ],
    )
    monkeypatch.setattr(sys, "argv", ["build_final.py"])
    build_final.main()
    assert len(load_jsonl(build_final.OUTPUT / "samples.jsonl")) == 2
    val_ids = (build_final.OUTPUT / "splits/val.txt").read_text().splitlines()
    assert val_ids == [new["sample_id"]]
    before = (build_final.OUTPUT / "samples.jsonl").read_bytes()
    build_final.main()
    assert (build_final.OUTPUT / "samples.jsonl").read_bytes() == before

    # Existing-label protection still fires after a reviewed text edit.
    changed = deepcopy(prepare_rewrite_inputs([old, new]))
    changed[0]["final_change"] += " and waving"
    write_jsonl(prepare_handoffs.REVIEWED, changed)
    with pytest.raises(ValueError, match="inspect its labels"):
        positives(clip=True)
    monkeypatch.setattr(sys, "argv", ["build_final.py"])
    with pytest.raises(ValueError, match="positive catalog is stale"):
        build_final.main()
