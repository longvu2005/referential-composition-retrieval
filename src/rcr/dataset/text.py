"""Parse and validate canonical Subject descriptions and target conditions."""

import re

from rcr.dataset.cases import subject_ids_for_case

SELECT_PREFIX = "Identify "
CHANGE_PREFIX = "then retrieve target images where "
SUBJECT_RE = re.compile(r"\bSubject\s+(\d+)\b")
TERMINAL_PUNCTUATION = ".;:!?"


def _field(output: dict, name: str) -> str:
    value = output.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    value = value.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be exactly one non-empty line.")
    return value


def _subject_ids(text: str) -> list[int]:
    return [int(x) for x in SUBJECT_RE.findall(text)]


def parse_selection_texts(final_desc: str, subject_ids: list[int]) -> list[str]:
    """Extract one referential description for each Subject in input order."""

    if not isinstance(final_desc, str) or not final_desc.startswith(SELECT_PREFIX):
        raise ValueError(f"final_desc must begin with {SELECT_PREFIX!r}.")

    body = final_desc.removeprefix(SELECT_PREFIX)
    parts = re.split(r"\s+and\s+(?=Subject\s+\d+\s+as\s+)", body)
    prefixes = [f"Subject {subject_id} as " for subject_id in subject_ids]

    if len(parts) != len(prefixes):
        raise ValueError(
            "final_desc must use `Subject N as <description>` for every Subject."
        )

    descriptions = []
    for part, prefix in zip(parts, prefixes, strict=True):
        if not part.startswith(prefix):
            raise ValueError(
                "final_desc must use `Subject N as <description>` for every Subject."
            )
        description = part[len(prefix) :].strip()
        if not description:
            raise ValueError(
                "final_desc must use `Subject N as <description>` for every Subject."
            )
        descriptions.append(description)

    return descriptions


def _validate_output(
    output: dict,
    subject_ids: list[int],
    required_change_ids: set[int],
) -> tuple[str, str]:
    if not isinstance(output, dict):
        raise ValueError("accepted text must be a JSON object.")
    if set(output) != {"final_desc", "final_change"}:
        raise ValueError(
            "accepted text must contain exactly final_desc and final_change."
        )

    final_desc = _field(output, "final_desc")
    final_change = _field(output, "final_change")

    if not final_change.startswith(CHANGE_PREFIX):
        raise ValueError(f"final_change must begin with {CHANGE_PREFIX!r}.")
    if final_desc.endswith(tuple(TERMINAL_PUNCTUATION)):
        raise ValueError("final_desc must not end with punctuation.")
    if final_change.endswith(tuple(TERMINAL_PUNCTUATION)):
        raise ValueError("final_change must not end with punctuation.")

    if _subject_ids(final_desc) != subject_ids:
        raise ValueError(
            "final_desc must mention every Subject exactly once in input order."
        )

    parse_selection_texts(final_desc, subject_ids)

    if set(_subject_ids(final_change)) != required_change_ids:
        raise ValueError(
            "final_change must mention exactly the Subjects required "
            "by the current structure."
        )

    return final_desc, final_change


def validate_review_output(case_type: str, output: dict) -> tuple[str, str]:
    subject_ids = subject_ids_for_case(case_type)
    return _validate_output(output, subject_ids, set(subject_ids))
