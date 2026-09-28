"""RCR case taxonomy."""

CASE_TYPES = ("INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL")
EDITABLE_CASE_TYPES = ("INDIVIDUAL", "DUAL", "RELATIONAL")
ONE_SUBJECT_CASES = {"INDIVIDUAL", "GROUP"}
CASE_SUBJECT_IDS = {
    "INDIVIDUAL": [1],
    "GROUP": [1],
    "DUAL": [1, 2],
    "RELATIONAL": [1, 2],
}


def subject_ids_for_case(case_type: str) -> list[int]:
    if case_type not in CASE_SUBJECT_IDS:
        raise ValueError(f"invalid case_type {case_type!r}")
    return CASE_SUBJECT_IDS[case_type].copy()


def infer_single_subject_case(identity_count: int) -> str:
    return "GROUP" if identity_count >= 2 else "INDIVIDUAL"
