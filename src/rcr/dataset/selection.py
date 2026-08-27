"""Dataset selection utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

JsonObject = dict[str, Any]


def split_by_qc(
    records: Iterable[JsonObject],
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Split samples into QC-passed and QC-failed groups."""

    selected = []
    failed = []

    for record in records:
        if record["annotation"]["qc"]["pass"]:
            selected.append(record)
        else:
            failed.append(record)

    return selected, failed
