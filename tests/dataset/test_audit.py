from copy import deepcopy
from types import SimpleNamespace

from rcr.dataset.audit import audit_detection, negative_exclusions, positive_conflicts


def _sample(sample_id, query, positives):
    return {
        "sample_id": sample_id,
        "query_image_id": query,
        "target_image_id": positives[0],
        "positive_image_ids": positives,
        "case_type": "GROUP",
        "subjects": [{"subject_id": 1, "identity_ids": ["a", "b"]}],
        "final_change": "Subject 1 is sitting",
    }


def test_conflicts_ignore_self_keep_positive_labels_and_exclude_only_negatives():
    samples = [_sample("s1", "q1", ["p", "q2", "extra"]), _sample("s2", "q2", ["p"])]
    before = deepcopy(samples)
    conflicts = positive_conflicts(samples)
    assert conflicts[0]["disputed_image_ids"] == ["extra"]
    assert negative_exclusions(samples) == {"s1": set(), "s2": {"extra"}}
    assert samples == before


def test_subject_role_mapping_and_condition_separate_conflict_groups():
    first = _sample("s1", "q1", ["p"])
    second = _sample("s2", "q2", ["other"])
    second["final_change"] = "Subject 1 is standing"
    assert positive_conflicts([first, second]) == []
    second["final_change"] = first["final_change"]
    second["subjects"] = [{"subject_id": 2, "identity_ids": ["a", "b"]}]
    assert positive_conflicts([first, second]) == []


def test_cache_coverage_distinguishes_partial_group_from_complete_detection():
    import torch

    sample = _sample("s", "q", ["p", "n"])
    data = SimpleNamespace(
        gallery_ids=["q", "p", "n"],
        samples_by_id={"s": sample},
        splits={"train": ["s"], "val": [], "test": []},
        images_by_id={i: {"path": f"train/{i}.jpg"} for i in ["q", "p", "n"]},
        gt_head_boxes_by_image={
            i: [
                {"identity_id": "a", "x": 1, "y": 1, "width": 1, "height": 1},
                {"identity_id": "b", "x": 5, "y": 1, "width": 1, "height": 1},
            ]
            for i in ["q", "p", "n"]
        },
        image_path=lambda _: None,
    )
    loaded = []

    def item(index):
        loaded.append(index)
        boxes = torch.tensor(
            [[[0, 0, 3, 10]], [[0, 0, 3, 10], [4, 0, 8, 10]], [[20, 20, 30, 30]]][
                index
            ],
            dtype=torch.float32,
        )
        return {
            "boxes_scene": boxes,
            "boxes_pixel": boxes,
            "identity_ids": ["ignored legacy label"],
        }

    cache = SimpleNamespace(
        image_ids=data.gallery_ids,
        cache_id="tiny",
        _load_item=item,
        validate_gallery=lambda ids: None,
        _scene_cache=None,
        encoder_metadata={},
    )
    result = audit_detection(data, cache)["splits"]["train"]["overall"]
    assert result["subjects_any_detected"] == 1
    assert (
        result["subjects_all_detected"] == result["query_all_identities_detected"] == 0
    )
    assert result["positive_targets_all_detected"] == 1
    assert len(loaded) == 3
