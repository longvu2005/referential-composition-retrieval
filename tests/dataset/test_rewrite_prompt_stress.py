"""Coverage checks for the live rewrite prompt stress fixture."""

from pathlib import Path

from rcr.utils.jsonl import load_jsonl

FIXTURE = Path("tests/fixtures/rewrite_prompt_stress.jsonl")

REQUIRED_RISK_TAGS = {
    "ambiguous_pronoun",
    "attachment",
    "between",
    "change_plus_relation",
    "compound_relation",
    "different",
    "disjunction",
    "either",
    "grammar_repair",
    "group_subject",
    "joint_pair_state",
    "left_right",
    "malformed_source",
    "missing_copula",
    "negation",
    "non_subject_person",
    "null_change",
    "number",
    "object_binding",
    "one_other",
    "ordinal",
    "output_format",
    "ownership",
    "pair_change",
    "passive_relation",
    "position_swap",
    "quantity",
    "relation_direction",
    "relation_only",
    "same_different",
    "subject_binding",
    "symmetric_relation",
    "tense_modality",
    "third_person_in_relation",
    "visible_text",
}


def test_stress_fixture_has_unique_ids_and_required_coverage() -> None:
    records = load_jsonl(FIXTURE)

    ids = [record["submission_id"] for record in records]
    assert len(ids) == len(set(ids))
    assert len(records) >= 30

    covered = {tag for record in records for tag in record["risk_tags"]}
    assert REQUIRED_RISK_TAGS <= covered


def test_stress_fixture_follows_rewrite_input_contract() -> None:
    records = load_jsonl(FIXTURE)

    for record in records:
        subject_ids = [subject["subject_id"] for subject in record["subjects"]]
        assert subject_ids == list(range(1, len(subject_ids) + 1))

        if record["case_type"] == "SINGLE":
            assert len(record["subjects"]) == 1
            assert record["pair_change"] is None
        elif record["case_type"] == "MULTI":
            assert len(record["subjects"]) >= 2
            assert record["pair_change"] is None
        else:
            assert record["case_type"] == "RELATIONAL"
            assert len(record["subjects"]) >= 2
            pair_change = record["pair_change"]
            assert pair_change is not None
            assert pair_change["subject_1_id"] in subject_ids
            assert pair_change["subject_2_id"] in subject_ids
