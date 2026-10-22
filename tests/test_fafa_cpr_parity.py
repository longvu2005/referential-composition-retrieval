"""CPU contracts shared with cpr_baseline_bench FAFA + Partial SetMatch."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from rcr.baselines.fafa_detector import filter_person_boxes, _split_batch
from rcr.baselines.fafa_matching import aggregate_people, partial_setmatch
from rcr.baselines.fafa_adapter import select_subject_members, subject_changes


def _reference(s, b):
    if not s.shape[1]:
        return b
    rows, cols = linear_sum_assignment(-s)
    return (s[rows, cols].sum() + (len(s) - len(rows)) * b) / len(s)


@pytest.mark.parametrize("k,n", [(1,1),(1,5),(2,1),(2,2),(2,6),(3,1),(3,2),(3,4),(5,3)])
def test_partial_setmatch_exact(k, n):
    rng = np.random.default_rng(42 + k + n)
    matrix = rng.uniform(-1, 1, (k, n))
    assert partial_setmatch(matrix, -1.0) == pytest.approx(_reference(matrix, -1.0))


def test_mean_penalty_not_minimum():
    s = np.array([[0.8], [0.6]])
    assert partial_setmatch(s, -1) == pytest.approx(-0.1)
    matrix = np.array([[0.81, 0.23], [0.20, 0.79]])
    assert partial_setmatch(matrix, -1) == pytest.approx(0.8)


def test_batch_aggregation_against_rectangular_reference():
    rng = np.random.default_rng(3)
    offsets = np.r_[0, np.cumsum([1, 2, 4, 1, 5, 3, 2])]
    for k in range(1, 8):
        scores = rng.uniform(-1, 1, (k, offsets[-1]))
        got = aggregate_people(scores, offsets, -1)
        expected = np.array([_reference(scores[:, a:b], -1)
                             for a, b in zip(offsets[:-1], offsets[1:])])
        np.testing.assert_allclose(got, expected, atol=1e-6)


def test_anchor_relative_margin_does_not_use_row_max():
    # Global Hungarian reserves person 0 for Subject 2; Subject 1 anchor=person 1.
    sim = np.array([[0.92, 0.88, 0.86], [0.99, 0.1, 0.1]])
    members = select_subject_members(sim, threshold=0.2, margin=0.03, max_members=5)
    assert members == [[1, 2], [0]]


def test_cpr_text_independent_and_relational():
    sample = {
        "subjects": [{"subject_id": 1}, {"subject_id": 2}],
        "final_change": (
            "then retrieve target images where Subject 1 is wearing red "
            "and Subject 2 is holding a bag"
        ),
    }
    assert subject_changes(sample) == ["is wearing red", "is holding a bag"]
    sample["final_change"] = (
        "then retrieve target images where Subject 1 stands behind Subject 2"
    )
    assert subject_changes(sample) == [
        "Subject 1 stands behind Subject 2", "Subject 1 stands behind Subject 2"
    ]


def test_detector_floor_ceil_nms_and_limits():
    cfg = {"box_threshold": .35, "min_box_size_px": 4, "nms_iou_threshold": .5,
           "max_persons_per_image": 10}
    boxes = [[.5,.5,.4,.4], [.51,.5,.4,.4], [.1,.1,.02,.02]]
    filtered, conf = filter_person_boxes(boxes, [.9,.8,.7], 100, 100, cfg)
    assert filtered.tolist() == [[30, 30, 70, 70]]
    assert conf.tolist() == pytest.approx([.9])


def test_detector_forward_splits_by_padding_ratio():
    import torch
    ts = [torch.zeros(3, 10, 10), torch.zeros(3, 10, 10), torch.zeros(3, 100, 10)]
    batches = list(_split_batch(ts, [1]*3, [1]*3, 1.3))
    assert [len(b[0]) for b in batches] == [2, 1]


def test_baseline_and_proposed_share_128_token_policy():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / "configs/fafa.yaml").read_text())
    proposed = yaml.safe_load((root / "configs/proposed.yaml").read_text())
    native = yaml.safe_load(
        (root / proposed["person_encoder"]["fafa_config"]).read_text()
    )
    assert baseline["checkpoint"]["max_txt_len"] == 128
    assert native["checkpoint"]["max_txt_len"] == 128
    assert proposed["person_encoder"]["fafa_config"] == "configs/fafa.yaml"
    assert baseline["checkpoint"]["path"] == native["checkpoint"]["path"]


def test_native_fafa_text_policy_matches_cpr_128():
    from rcr.baselines.fafa import text_policy

    assert text_policy({"checkpoint": {"max_txt_len": 128}}) == {
        "max_txt_len": 128, "max_words": 128,
    }
