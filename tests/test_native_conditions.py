"""Public Subject anchoring for the proposed frozen FAFA query branch."""

from copy import deepcopy

import pytest

from rcr.proposed.cache.native import query_key, subject_conditions


@pytest.mark.parametrize("conjunction", ["and", "while"])
def test_every_subject_keeps_the_complete_condition(conjunction):
    sample = {
        "subjects": [{"subject_id": 1}, {"subject_id": 2}],
        "final_change": (
            "then retrieve target images where Subject 1 is wearing red "
            f"{conjunction} Subject 2 is holding a bag"
        ),
    }
    assert subject_conditions(sample) == [
        "the reference person is wearing red "
        f"{conjunction} the other referenced person (Subject 2) is holding a bag",
        "the other referenced person (Subject 1) is wearing red "
        f"{conjunction} the reference person is holding a bag",
    ]


@pytest.mark.parametrize(
    "ids,condition,expected",
    [
        ([1], "Subject 1 is sitting.", ["the reference person is sitting."]),
        (
            [1],
            "the members of Subject 1 are standing together.",
            ["the reference person is standing together."],
        ),
        (
            [2, 1],
            "members of Subject 1 were seated and Subject 2 has joined Subject 1.",
            [
                "the other referenced person (Subject 1) was seated "
                "and the reference person has joined "
                "the other referenced person (Subject 1).",
                "the reference person was seated "
                "and the other referenced person (Subject 2) has joined "
                "the reference person.",
            ],
        ),
        (
            [1, 2],
            "Subject 1 stands behind Subject 2 and Subject 2 looks at Subject 1.",
            [
                "the reference person stands behind "
                "the other referenced person (Subject 2) "
                "and the other referenced person (Subject 2) looks at "
                "the reference person.",
                "the other referenced person (Subject 1) stands behind "
                "the reference person and the reference person looks at "
                "the other referenced person (Subject 1).",
            ],
        ),
        (
            [1, 2],
            "Subject 1 is beside a man and Subject 2 is facing the same man.",
            [
                "the reference person is beside a man "
                "and the other referenced person (Subject 2) is facing the same man.",
                "the other referenced person (Subject 1) is beside a man "
                "and the reference person is facing the same man.",
            ],
        ),
        (
            [10, 1],
            "Subject 1 has a bag and Subject 10 is seated.",
            [
                "the other referenced person (Subject 1) has a bag "
                "and the reference person is seated.",
                "the reference person has a bag "
                "and the other referenced person (Subject 10) is seated.",
            ],
        ),
    ],
)
def test_anchor_preserves_relations_group_agreement_and_subject_order(
    ids, condition, expected
):
    sample = {
        "subjects": [{"subject_id": sid} for sid in ids],
        "final_change": "then retrieve target images where " + condition,
    }
    original = deepcopy(sample)
    assert subject_conditions(sample) == expected
    assert sample == original


def test_conditions_and_cache_keys_use_no_private_labels():
    sample = {
        "query_image_id": "query",
        "subjects": [{"subject_id": 1}, {"subject_id": 2}],
        "final_change": "Subject 1 is seated and Subject 2 is standing",
    }
    labeled = deepcopy(sample)
    labeled["case_type"] = "DUAL"
    for subject in labeled["subjects"]:
        subject.update(identity_ids=["private-id"], member_count=5)
    assert subject_conditions(labeled) == subject_conditions(sample)
    assert query_key(labeled) == query_key(sample)
