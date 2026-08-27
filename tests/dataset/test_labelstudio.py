"""Tests for the offline Label Studio handoff adapter."""

import pytest

import labelstudio.positives.collect as positive_collect
from labelstudio.common import load_box_index, local_image_url
from labelstudio.positives.collect import aggregate
from labelstudio.positives.prepare import build_tasks
from labelstudio.review.collect import parse_task
from labelstudio.review.prepare import build_task
from rcr.utils.jsonl import load_jsonl, write_jsonl

SUBJECTS = [{"subject_id": 1, "identity_ids": ["7"]}]
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
    ]
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


def test_local_image_url_uses_repo_image_root() -> None:
    url = "/data/local-files/?d=Amazing/Datasets/PIPA/images/train/1_1.jpg"
    assert local_image_url(url) == "/data/local-files/?d=train/1_1.jpg"


def test_review_task_keeps_empty_rewrite_for_human() -> None:
    row = {
        "submission_id": "s1",
        "case_type": "SINGLE",
        "query_image_id": "1_1",
        "query_image_url": "/data/local-files/?d=train/1_1.jpg",
        "target_image_id": "1_1",
        "target_image_url": "/data/local-files/?d=train/1_1.jpg",
        "subjects": [
            {
                "subject_id": 1,
                "identity_ids": ["7"],
                "description": "the man",
                "change": "is smiling",
            }
        ],
        "pair_change": None,
        "final_desc": None,
        "final_change": None,
    }

    task = build_task(row, BOXES)

    assert task["data"]["final_desc"] == ""
    assert task["data"]["case_choices"] == [
        {"value": "SINGLE", "selected": True},
        {"value": "MULTI", "selected": False},
        {"value": "RELATIONAL", "selected": False},
    ]
    assert "Subject 1" in task["data"]["subjects_html"]
    assert "border:3px solid" in task["data"]["images_html"]


def test_review_collect_reads_both_textareas() -> None:
    task = {
        "data": {"task_key": "s1"},
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "case_type",
                        "value": {"choices": ["RELATIONAL"]},
                    },
                    {
                        "from_name": "final_desc",
                        "value": {"text": ["Identify Subject 1 as the man"]},
                    },
                    {
                        "from_name": "final_change",
                        "value": {
                            "text": [
                                "then retrieve target images where Subject 1 is smiling"
                            ]
                        },
                    },
                ]
            }
        ],
    }

    source = {
        "submission_id": "s1",
        "subjects": [{"subject_id": 1, "change": "is smiling"}],
        "pair_change": None,
    }
    row = parse_task(task, source)

    assert row is not None
    assert row["submission_id"] == "s1"
    assert row["case_type"] == "RELATIONAL"


def test_review_collect_falls_back_to_existing_case() -> None:
    task = {
        "data": {"task_key": "s1"},
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "final_desc",
                        "value": {"text": ["Identify Subject 1 as the man"]},
                    },
                    {
                        "from_name": "final_change",
                        "value": {
                            "text": [
                                "then retrieve target images where Subject 1 is smiling"
                            ]
                        },
                    },
                ]
            }
        ],
    }
    source = {
        "submission_id": "s1",
        "case_type": "SINGLE",
        "subjects": [{"subject_id": 1, "change": "is smiling"}],
        "pair_change": None,
    }

    row = parse_task(task, source)

    assert row is not None
    assert row["case_type"] == "SINGLE"


def test_positive_prepare_groups_candidates_by_ten() -> None:
    candidates = [
        {
            "image_id": "seed",
            "image_url": "/data/local-files/?d=train/seed.jpg",
            "is_seed": True,
            "subject_boxes": [],
        }
    ]
    candidates.extend(
        {
            "image_id": f"c{i}",
            "image_url": f"/data/local-files/?d=train/c{i}.jpg",
            "is_seed": False,
            "subject_boxes": [],
        }
        for i in range(23)
    )
    row = {
        "submission_id": "s1",
        "query_image_id": "1_1",
        "query_image_url": "/data/local-files/?d=train/1_1.jpg",
        "subjects": SUBJECTS,
        "final_desc": "Identify Subject 1 as the man",
        "final_change": "then retrieve target images where Subject 1 is smiling",
        "candidates": candidates,
    }

    tasks = build_tasks(row, BOXES)

    assert [len(task["data"]["candidate_choices"]) for task in tasks] == [10, 10, 3]
    assert [task["data"]["task_key"] for task in tasks] == [
        "s1::000",
        "s1::001",
        "s1::002",
    ]


def test_positive_collect_waits_for_all_groups() -> None:
    inputs = [
        {
            "submission_id": "s1",
            "seed_target_image_id": "seed",
            "candidates": [
                {"image_id": "seed", "is_seed": True},
                *[{"image_id": f"c{i}", "is_seed": False} for i in range(11)],
            ],
        }
    ]
    task = {
        "data": {"submission_id": "s1", "group_index": 0, "group_count": 2},
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "positives",
                        "value": {"choices": ["c0"]},
                    }
                ]
            }
        ],
    }

    assert aggregate([task], inputs) == []

    second = {
        "data": {"submission_id": "s1", "group_index": 1, "group_count": 2},
        "annotations": [{"result": []}],
    }
    assert aggregate([task, second], inputs) == [
        {"submission_id": "s1", "target_image_ids": ["seed", "c0"]}
    ]


def test_positive_collect_rejects_choice_outside_canonical_group() -> None:
    inputs = [
        {
            "submission_id": "s1",
            "seed_target_image_id": "seed",
            "candidates": [
                {"image_id": "seed", "is_seed": True},
                {"image_id": "a", "is_seed": False},
            ],
        }
    ]
    task = {
        "data": {"submission_id": "s1", "group_index": 0, "group_count": 1},
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "positives",
                        "value": {"choices": ["not-a-candidate"]},
                    }
                ]
            }
        ],
    }

    with pytest.raises(ValueError, match="invalid positive choice"):
        aggregate([task], inputs)


def test_positive_collect_handles_seed_only_without_export(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "positive_set_input.jsonl"
    export_path = tmp_path / "missing_export.json"
    output_path = tmp_path / "positive_sets.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "submission_id": "s1",
                "seed_target_image_id": "seed",
                "candidates": [{"image_id": "seed", "is_seed": True}],
            }
        ],
    )
    monkeypatch.setattr(positive_collect, "INPUT", input_path)
    monkeypatch.setattr(positive_collect, "EXPORT", export_path)
    monkeypatch.setattr(positive_collect, "OUTPUT", output_path)

    positive_collect.main()

    assert load_jsonl(output_path) == [
        {"submission_id": "s1", "target_image_ids": ["seed"]}
    ]
