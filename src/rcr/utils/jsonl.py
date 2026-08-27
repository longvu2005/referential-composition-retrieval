"""Small JSONL helpers used across the project."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[JsonObject]:
    """Iterate over non-empty JSONL records."""

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[JsonObject]:
    """Load all records from a JSONL file."""

    return list(iter_jsonl(path))


def write_jsonl(
    path: str | Path,
    records: Iterable[JsonObject],
) -> None:
    """Write records to a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
