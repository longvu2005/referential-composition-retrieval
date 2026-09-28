"""Minimal local UI for RCR Full Positive selection."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from rcr.dataset.cases import CASE_TYPES
from rcr.dataset.positives import validate_positive_set as validate_submission
from rcr.utils.images import image_relative_path as _image_relative_path
from rcr.utils.jsonl import load_jsonl, write_jsonl

INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
OUTPUT = Path("dataset/data/work/positives/positive_sets.jsonl")
IMAGE_ROOT = Path("dataset/data/raw/images")
HTML = Path(__file__).with_name("app.html")


def build_task_payload(
    row: dict,
    positive: dict | None = None,
) -> dict:
    """Build a task; stored positives are validated on load and browser save."""

    selected = (
        positive["positive_image_ids"]
        if positive is not None
        else [row["seed_target_image_id"]]
    )

    candidates = []
    for candidate in row["candidates"]:
        image_id = candidate["image_id"]
        candidates.append(
            {
                "image_id": image_id,
                "image_url": (
                    "/api/image?path="
                    f"{_image_relative_path(candidate['image_url']).as_posix()}"
                ),
                "is_seed": candidate.get("is_seed") is True,
                "boxes": candidate.get("subject_boxes", []),
            }
        )

    return {
        "sample_id": row["sample_id"],
        "case_type": row["case_type"],
        "subjects": row["subjects"],
        "query": {
            "image_id": row["query_image_id"],
            "image_url": (
                "/api/image?path="
                f"{_image_relative_path(row['query_image_url']).as_posix()}"
            ),
            "boxes": [],
        },
        "final_desc": row["final_desc"],
        "final_change": row["final_change"],
        "candidates": candidates,
        "positive_image_ids": selected,
    }


class PositiveState:
    def __init__(self) -> None:
        self.rows = load_jsonl(INPUT)
        self.rows_by_id = {row["sample_id"]: row for row in self.rows}
        self.index_by_id = {row["sample_id"]: i for i, row in enumerate(self.rows)}
        if len(self.rows_by_id) != len(self.rows):
            raise ValueError("positive_set_input.jsonl contains duplicate sample_id")

        # Validate catalog structure once; task rendering uses these same rows.
        for row in self.rows:
            candidate_ids = [item["image_id"] for item in row["candidates"]]
            seed_ids = [
                item["image_id"] for item in row["candidates"] if item.get("is_seed")
            ]
            seed_id = row["seed_target_image_id"]
            if (
                not candidate_ids
                or len(candidate_ids) != len(set(candidate_ids))
                or candidate_ids[0] != seed_id
                or seed_ids != [seed_id]
            ):
                raise ValueError(
                    f"{row['sample_id']}: candidates must be unique "
                    "with the canonical seed first and marked exactly once"
                )

        existing = load_jsonl(OUTPUT) if OUTPUT.exists() else []
        if len({row["sample_id"] for row in existing}) != len(existing):
            raise ValueError("positive_sets.jsonl contains duplicate sample_id")

        unknown = [
            row["sample_id"]
            for row in existing
            if row["sample_id"] not in self.rows_by_id
        ]
        if unknown:
            raise ValueError(
                "positive_sets.jsonl contains IDs outside positive_set_input.jsonl: "
                f"{unknown[0]}"
            )

        self.positives = {
            row["sample_id"]: validate_submission(
                self.rows_by_id[row["sample_id"]], row
            )
            for row in existing
        }
        self.lock = threading.Lock()

    def _meta(self, index: int, row: dict) -> dict:
        return {
            "index": index,
            "sample_id": row["sample_id"],
            "case_type": row["case_type"],
            "completed": row["sample_id"] in self.positives,
        }

    def tasks(self) -> dict:
        """Return lightweight task metadata for navigation and filtering."""

        tasks = [self._meta(index, row) for index, row in enumerate(self.rows)]
        completed = sum(item["completed"] for item in tasks)
        return {
            "total": len(tasks),
            "completed": completed,
            "pending": len(tasks) - completed,
            "case_types": list(CASE_TYPES),
            "tasks": tasks,
        }

    def task(self, index: int | None = None, sample_id: str | None = None) -> dict:
        """Load a task by its sample ID, with index kept for navigation."""

        if sample_id is not None:
            row = self.rows_by_id.get(sample_id)
            if row is None:
                raise ValueError("unknown sample_id")
            index = self.index_by_id[sample_id]
        else:
            index = 0 if index is None else index
            if index < 0 or index >= len(self.rows):
                raise ValueError("task index out of range")
            row = self.rows[index]

        positive = self.positives.get(row["sample_id"])
        summary = self.tasks()
        return {
            "index": index,
            "total": summary["total"],
            "completed": summary["completed"],
            "pending": summary["pending"],
            "is_completed": positive is not None,
            "task": build_task_payload(row, positive),
        }

    def save(self, payload: dict) -> dict:
        sample_id = payload.get("sample_id")
        source = self.rows_by_id.get(sample_id)
        if source is None:
            raise ValueError("unknown sample_id")
        row = validate_submission(source, payload)

        with self.lock:
            self.positives[sample_id] = row
            temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
            write_jsonl(
                temporary,
                [
                    self.positives[source_row["sample_id"]]
                    for source_row in self.rows
                    if source_row["sample_id"] in self.positives
                ],
            )
            os.replace(temporary, OUTPUT)
        return row


class PositiveHandler(BaseHTTPRequestHandler):
    state: PositiveState

    def _json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, error: Exception) -> None:
        self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = HTML.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/api/tasks":
                self._json(self.state.tasks())
                return

            if parsed.path == "/api/task":
                params = parse_qs(parsed.query)
                sample_id = params.get("id", [None])[0]
                index = None
                if sample_id is None:
                    index = int(params.get("index", ["0"])[0])
                self._json(self.state.task(index=index, sample_id=sample_id))
                return

            if parsed.path == "/api/image":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                relative = Path(unquote(raw_path))
                if not raw_path or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe image path")
                image_path = IMAGE_ROOT / relative
                data = image_path.read_bytes()
                content_type = (
                    mimetypes.guess_type(image_path.name)[0]
                    or "application/octet-stream"
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:  # local annotation server: surface useful errors
            self._error(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/positive":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            row = self.state.save(payload)
            self._json({"positive": row, "summary": self.state.tasks()})
        except Exception as error:  # local annotation server: surface useful errors
            self._error(error)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RCR positive UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    state = PositiveState()
    handler = type("BoundPositiveHandler", (PositiveHandler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"RCR Full Positive selection: http://{args.host}:{args.port}")
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
