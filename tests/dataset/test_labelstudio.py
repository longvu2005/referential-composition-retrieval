"""Tests for shared annotation metadata and the Full Positive local UI."""

import threading

import pytest

import labelstudio.positives.app as positive_app
from labelstudio.common import load_box_index
from labelstudio.positives.app import (
    PositiveState,
    _image_relative_path,
    build_task_payload,
    validate_submission,
)
from rcr.common.io import load_jsonl, write_jsonl

BOXES = {
    "1_1": [
        {
            "image_id": "1_1",
            "label": "7",
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.4,
        }
    ],
    "1_2": [
        {
            "image_id": "1_2",
            "label": "7",
            "x": 0.2,
            "y": 0.1,
            "width": 0.2,
            "height": 0.4,
        }
    ],
    "1_3": [
        {
            "image_id": "1_3",
            "label": "7",
            "x": 0.3,
            "y": 0.2,
            "width": 0.2,
            "height": 0.3,
        }
    ],
}


def _source(sample_id: str = "s1") -> dict:
    return {
        "sample_id": sample_id,
        "case_type": "INDIVIDUAL",
        "query_image_id": "1_1",
        "query_image_url": (
            "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_1.jpg"
        ),
        "seed_target_image_id": "1_2",
        "subjects": [{"subject_id": 1, "identity_ids": ["7"]}],
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
        "candidates": [
            {
                "image_id": "1_2",
                "image_url": "/data/local-files/?d=train/1_2.jpg",
                "is_seed": True,
                "subject_boxes": [
                    {
                        "subject_id": 1,
                        "identity_id": "7",
                        "x": 20,
                        "y": 10,
                        "width": 20,
                        "height": 40,
                    }
                ],
            },
            {
                "image_id": "1_3",
                "image_url": "/data/local-files/?d=train/1_3.jpg",
                "is_seed": False,
                "subject_boxes": [
                    {
                        "subject_id": 1,
                        "identity_id": "7",
                        "x": 30,
                        "y": 20,
                        "width": 20,
                        "height": 30,
                    }
                ],
            },
        ],
        "positive_image_ids": ["1_2"],
    }


def test_load_box_index_recovers_boxes_missing_from_pair_data(tmp_path) -> None:
    pair_path = tmp_path / "pair_data.json"
    index_path = tmp_path / "index.txt"
    pair_path.write_text(
        '{"images":[{"image_id":"1_1"}],"boxes":['
        '{"image_id":"1_1","label":"7","x":0.1,"y":0.2,'
        '"width":0.3,"height":0.4}]}',
        encoding="utf-8",
    )
    index_path.write_text(
        "1 1 10 20 30 40 7 1\n1 1 50 20 20 30 8 1\n",
        encoding="utf-8",
    )

    boxes = load_box_index(pair_path, index_path)

    assert [box["label"] for box in boxes["1_1"]] == ["7", "8"]
    assert boxes["1_1"][1] == {
        "image_id": "1_1",
        "label": "8",
        "x": 0.5,
        "y": 0.2,
        "width": 0.2,
        "height": 0.3,
    }


def test_positive_task_payload_uses_handoff_candidate_boxes() -> None:
    task = build_task_payload(_source())

    assert task["positive_image_ids"] == ["1_2"]
    assert task["query"]["boxes"] == []
    assert task["candidates"][0]["is_seed"] is True
    assert task["candidates"][1]["boxes"][0] == {
        "subject_id": 1,
        "identity_id": "7",
        "x": 30,
        "y": 20,
        "width": 20,
        "height": 30,
    }


def test_positive_submission_requires_seed_and_rejects_unknown_candidates() -> None:
    source = _source()

    with pytest.raises(ValueError, match="seed target must remain selected"):
        validate_submission(
            source,
            {"sample_id": "s1", "positive_image_ids": ["1_3"]},
        )

    with pytest.raises(ValueError, match="invalid positive_image_id"):
        validate_submission(
            source,
            {"sample_id": "s1", "positive_image_ids": ["1_2", "unknown"]},
        )


def test_positive_submission_uses_canonical_candidate_order() -> None:
    row = validate_submission(
        _source(),
        {"sample_id": "s1", "positive_image_ids": ["1_3", "1_2"]},
    )

    assert row == {
        "sample_id": "s1",
        "positive_image_ids": ["1_2", "1_3"],
    }


def test_positive_state_loads_cumulative_catalog(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "positive_set_input.jsonl"
    output_path = tmp_path / "positive_sets.jsonl"
    write_jsonl(input_path, [_source("s1"), _source("s2")])
    write_jsonl(
        output_path,
        [{"sample_id": "s1", "positive_image_ids": ["1_2", "1_3"]}],
    )
    monkeypatch.setattr(positive_app, "INPUT", input_path)
    monkeypatch.setattr(positive_app, "OUTPUT", output_path)
    state = PositiveState()

    assert state.tasks()["completed"] == 1
    assert state.task(sample_id="s1")["is_completed"] is True
    assert state.task(sample_id="s1")["task"]["positive_image_ids"] == [
        "1_2",
        "1_3",
    ]


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing_seed", "extra_seed", "seed_not_first"]
)
def test_positive_state_rejects_invalid_catalog(tmp_path, monkeypatch, problem):
    row = _source()
    if problem == "duplicate":
        row["candidates"].append(dict(row["candidates"][1]))
    elif problem == "missing_seed":
        row["candidates"][0]["is_seed"] = False
    elif problem == "extra_seed":
        row["candidates"][1]["is_seed"] = True
    else:
        row["candidates"].reverse()

    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, [row])
    monkeypatch.setattr(positive_app, "INPUT", input_path)
    monkeypatch.setattr(positive_app, "OUTPUT", tmp_path / "positive_sets.jsonl")
    with pytest.raises(ValueError, match="candidates must be unique"):
        PositiveState()


def test_positive_state_tasks_keeps_completed_tasks_visible() -> None:
    first = _source("s1")
    second = _source("s2")
    state = PositiveState.__new__(PositiveState)
    state.rows = [first, second]
    state.positives = {"s1": {"sample_id": "s1", "positive_image_ids": ["1_2"]}}

    summary = state.tasks()

    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["pending"] == 1
    assert summary["tasks"][0]["completed"] is True
    assert summary["tasks"][1]["completed"] is False


def test_positive_state_save_writes_catalog_order(tmp_path, monkeypatch) -> None:
    output = tmp_path / "positive_sets.jsonl"
    first = _source("s1")
    second = _source("s2")
    state = PositiveState.__new__(PositiveState)
    state.rows = [first, second]
    state.rows_by_id = {"s1": first, "s2": second}
    state.positives = {}
    state.lock = threading.Lock()
    monkeypatch.setattr(positive_app, "OUTPUT", output)

    state.save({"sample_id": "s2", "positive_image_ids": ["1_2"]})
    state.save({"sample_id": "s1", "positive_image_ids": ["1_2", "1_3"]})

    assert load_jsonl(output) == [
        {"sample_id": "s1", "positive_image_ids": ["1_2", "1_3"]},
        {"sample_id": "s2", "positive_image_ids": ["1_2"]},
    ]


def test_positive_image_relative_path_is_safe() -> None:
    assert (
        _image_relative_path(
            "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_1.jpg"
        ).as_posix()
        == "train/1_1.jpg"
    )

    with pytest.raises(ValueError, match="unsafe image path"):
        _image_relative_path("/images/../secret.jpg")
