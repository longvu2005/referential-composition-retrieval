"""Full Positive auditing distinguishes validity, review scope and semantics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from labelstudio.positives.app import build_task_payload
from rcr.dataset.positive_audit import audit_full_positives, full_positive_review_queue
from rcr.dataset.positives import validate_positive_set


def _data():
    images = ["q", "p", "negative", "missing", "wrong", "outside", "left"]
    row = {
        "sample_id": "s",
        "case_type": "GROUP",
        "query_image_id": "q",
        "target_image_id": "p",
        "positive_image_ids": ["p", "outside"],
        "subjects": [{"subject_id": 1, "identity_ids": ["a", "b"]}],
        "final_desc": "Identify Subject 1 as the two people in red",
        "final_change": "then retrieve target images where Subject 1 is sitting",
    }
    heads = [
        {"identity_id": i, "x": n * 10, "y": 0, "width": 2, "height": 2}
        for n, i in enumerate(["a", "b"])
    ]
    image_splits = {"outside": "val", "left": "leftover"}
    return SimpleNamespace(
        gallery_ids=images,
        images_by_id={
            i: {"path": f"{image_splits.get(i, 'train')}/{i}.jpg"} for i in images
        },
        samples_by_id={"s": row},
        splits={"train": ["s"], "val": [], "test": []},
        gt_head_boxes_by_image={
            i: (heads[:1] if i == "wrong" else heads) for i in images
        },
    )


def _row(report):
    return report["splits"]["train"]["queries"][0]


def _source(data, ids=("p", "negative", "missing")):
    sample = data.samples_by_id["s"]
    return {
        **{
            k: sample[k]
            for k in (
                "sample_id",
                "case_type",
                "query_image_id",
                "subjects",
                "final_desc",
                "final_change",
            )
        },
        "seed_target_image_id": "p",
        "candidates": [{"image_id": i} for i in ids],
    }


def test_unknown_scope_and_queue_cover_all_id_candidates_without_editing_labels():
    data = _data()
    before = deepcopy(vars(data))
    report = audit_full_positives(data)
    row = _row(report)
    assert row["annotation_issues"] == []
    assert row["id_positive_image_ids"] == ["p", "negative", "missing"]
    assert row["full_positive_image_ids"] == ["p"]
    assert row["out_of_split_positive_image_ids"] == ["outside"]
    assert row["scope_status"] == "unknown"
    assert row["unconfirmed_negative_image_ids"] == ["negative", "missing"]
    assert report["semantic_completeness"] == "not_established_by_code"
    queue = full_positive_review_queue(data, report)
    assert [c["image_id"] for c in queue[0]["candidates"]] == [
        "p",
        "negative",
        "missing",
    ]
    assert queue[0]["positive_image_ids"] == ["p"]
    validate_positive_set(queue[0], {"sample_id": "s", "positive_image_ids": ["p"]})
    assert vars(data) == before


def test_partial_scope_exposes_only_unconfirmed_negatives():
    data = _data()
    source = _source(data, ("p", "negative"))
    row = _row(
        audit_full_positives(
            data, [source], [{"sample_id": "s", "positive_image_ids": ["p"]}]
        )
    )
    assert row["scope_status"] == "partial_catalog"
    assert row["catalog_negative_image_ids"] == ["negative"]
    assert row["unconfirmed_negative_image_ids"] == ["missing"]
    assert data.samples_by_id["s"]["positive_image_ids"] == ["p", "outside"]


def test_matching_complete_catalog_is_scope_evidence_not_semantic_certification():
    data = _data()
    report = audit_full_positives(
        data, [_source(data)], [{"sample_id": "s", "positive_image_ids": ["p"]}]
    )
    assert _row(report)["scope_status"] == "complete_catalog"
    assert _row(report)["unconfirmed_negative_image_ids"] == []
    assert full_positive_review_queue(data, report) == []
    assert report["semantic_completeness"] == "not_established_by_code"


@pytest.mark.parametrize(
    "change", ["text", "subjects", "decision", "duplicate_candidates"]
)
def test_stale_context_or_invalid_decision_cannot_certify_scope(change):
    data = _data()
    source = deepcopy(_source(data))
    decision = {"sample_id": "s", "positive_image_ids": ["p"]}
    if change == "text":
        source["final_change"] += " changed"
    elif change == "subjects":
        source["subjects"][0]["identity_ids"] = ["different"]
    elif change == "decision":
        decision["positive_image_ids"].append("negative")
    else:
        source["candidates"].append({"image_id": "p"})
    row = _row(audit_full_positives(data, [source], [decision]))
    assert row["scope_status"] == "stale"
    assert row["review_source_issues"]
    assert row["catalog_negative_image_ids"] == []
    assert row["unconfirmed_negative_image_ids"] == ["negative", "missing"]


def test_catalog_without_a_submitted_decision_does_not_establish_review_scope():
    data = _data()
    assert (
        _row(audit_full_positives(data, [_source(data)]))["scope_status"] == "unknown"
    )
    source = _source(data)
    source["final_change"] = "stale text"
    assert _row(audit_full_positives(data, [source]))["scope_status"] == "stale"


@pytest.mark.parametrize(
    "positives,issue",
    [
        (["p", "p"], "duplicate_positive_ids"),
        (["p", "q"], "self_positive"),
        (["p", "wrong"], "positive_missing_required_identities"),
        ([], "empty_split_positive_set"),
        (["negative"], "missing_seed_positive"),
        (["p", "not_registered"], "unregistered_positive_ids"),
    ],
)
def test_structural_errors_are_reported_without_automatic_correction(positives, issue):
    data = _data()
    data.samples_by_id["s"]["positive_image_ids"] = list(positives)
    report = audit_full_positives(data)
    assert issue in _row(report)["annotation_issues"]
    assert full_positive_review_queue(data, report) == []
    assert data.samples_by_id["s"]["positive_image_ids"] == positives


def test_equivalent_query_conflict_is_reviewed_even_with_complete_catalogs():
    data = _data()
    second = {
        **deepcopy(data.samples_by_id["s"]),
        "sample_id": "s2",
        "positive_image_ids": ["p", "negative"],
    }
    data.samples_by_id["s2"] = second
    data.splits["train"].append("s2")
    first_source = _source(data)
    second_source = {**first_source, "sample_id": "s2"}
    report = audit_full_positives(
        data,
        [first_source, second_source],
        [
            {"sample_id": "s", "positive_image_ids": ["p"]},
            {"sample_id": "s2", "positive_image_ids": ["p", "negative"]},
        ],
    )
    assert report["splits"]["train"]["summary"]["conflict_groups"] == 1
    assert _row(report)["disputed_image_ids"] == ["negative"]
    queue = full_positive_review_queue(data, report)
    assert len(queue) == 2
    assert queue[1]["positive_image_ids"] == ["p", "negative"]
    assert build_task_payload(queue[1])["positive_image_ids"] == ["p", "negative"]
