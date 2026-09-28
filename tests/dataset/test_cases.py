"""Tests for the canonical RCR case taxonomy."""

import pytest

from rcr.dataset.cases import (
    CASE_TYPES,
    infer_single_subject_case,
    subject_ids_for_case,
)


def test_case_types_and_subject_topology() -> None:
    assert CASE_TYPES == ("INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL")
    assert subject_ids_for_case("INDIVIDUAL") == [1]
    assert subject_ids_for_case("GROUP") == [1]
    assert subject_ids_for_case("DUAL") == [1, 2]
    assert subject_ids_for_case("RELATIONAL") == [1, 2]


def test_single_subject_case_is_derived_from_identity_count() -> None:
    assert infer_single_subject_case(0) == "INDIVIDUAL"
    assert infer_single_subject_case(1) == "INDIVIDUAL"
    assert infer_single_subject_case(2) == "GROUP"
    assert infer_single_subject_case(5) == "GROUP"


def test_unknown_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid case_type"):
        subject_ids_for_case("SINGLE")
