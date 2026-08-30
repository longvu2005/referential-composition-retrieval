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

from labelstudio.common import load_box_index
from rcr.utils.jsonl import load_jsonl, write_jsonl

INPUT = Path("dataset/data/work/positives/positive_set_input.jsonl")
OUTPUT = Path("dataset/data/work/positives/positive_sets.jsonl")
IMAGE_ROOT = Path("dataset/data/raw/images")
HTML = Path(__file__).with_name("app.html")


def _image_relative_path(url: str) -> Path:
    """Map PIPA/local-files URLs to split-relative image paths."""

    parsed = urlparse(url)
    value = parse_qs(parsed.query).get("d", [parsed.path])[0]
    value = unquote(value).replace("\\", "/")
    for marker in ("/PIPA/images/", "/images/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = value.lstrip("/")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image path {url!r}")
    return path


def _subject_boxes(
    row: dict,
    image_id: str,
    boxes_by_image: dict[str, list[dict]],
) -> list[dict]:
    subject_by_identity = {
        identity_id: subject["subject_id"]
        for subject in row["subjects"]
        for identity_id in subject["identity_ids"]
    }
    return [
        {
            "subject_id": subject_by_identity[box["label"]],
            "x": 100 * box["x"],
            "y": 100 * box["y"],
            "width": 100 * box["width"],
            "height": 100 * box["height"],
        }
        for box in boxes_by_image.get(image_id, [])
        if box["label"] in subject_by_identity
    ]


def _candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{row['submission_id']}: candidates are required")

    candidate_ids = [candidate.get("image_id") for candidate in candidates]
    if any(not isinstance(image_id, str) or not image_id for image_id in candidate_ids):
        raise ValueError(f"{row['submission_id']}: invalid candidate image_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{row['submission_id']}: duplicate candidate image_id")

    seed_ids = [
        candidate["image_id"] for candidate in candidates if candidate.get("is_seed")
    ]
    seed_id = row.get("seed_target_image_id")
    if seed_ids != [seed_id]:
        raise ValueError(f"{row['submission_id']}: expected exactly one canonical seed")
    if candidate_ids[0] != seed_id:
        raise ValueError(f"{row['submission_id']}: seed target must be first")
    return candidate_ids


def validate_submission(source: dict, payload: dict) -> dict:
    """Validate one browser submission and return the canonical positive row."""

    submission_id = source["submission_id"]
    if payload.get("submission_id") != submission_id:
        raise ValueError(f"{submission_id}: submission_id mismatch")

    candidate_ids = _candidate_ids(source)
    target_ids = payload.get("target_image_ids")
    if not isinstance(target_ids, list) or any(
        not isinstance(image_id, str) or not image_id for image_id in target_ids
    ):
        raise ValueError(f"{submission_id}: target_image_ids must be a string list")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"{submission_id}: duplicate target_image_id")

    invalid = set(target_ids) - set(candidate_ids)
    if invalid:
        raise ValueError(
            f"{submission_id}: invalid target_image_id {sorted(invalid)[0]}"
        )

    seed_id = source["seed_target_image_id"]
    if seed_id not in target_ids:
        raise ValueError(f"{submission_id}: seed target must remain selected")

    selected = set(target_ids)
    return {
        "submission_id": submission_id,
        "target_image_ids": [
            image_id for image_id in candidate_ids if image_id in selected
        ],
    }


def build_task_payload(
    row: dict,
    boxes_by_image: dict[str, list[dict]],
    positive: dict | None = None,
) -> dict:
    """Build one browser task from canonical positive-set input."""

    candidate_ids = _candidate_ids(row)
    selected = (
        validate_submission(row, positive)["target_image_ids"]
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
                "boxes": _subject_boxes(row, image_id, boxes_by_image),
            }
        )

    if [candidate["image_id"] for candidate in candidates] != candidate_ids:
        raise ValueError(
            f"{row['submission_id']}: candidate order changed unexpectedly"
        )

    return {
        "submission_id": row["submission_id"],
        "case_type": row["case_type"],
        "subjects": row["subjects"],
        "query": {
            "image_id": row["query_image_id"],
            "image_url": (
                "/api/image?path="
                f"{_image_relative_path(row['query_image_url']).as_posix()}"
            ),
            "boxes": _subject_boxes(row, row["query_image_id"], boxes_by_image),
        },
        "final_desc": row["final_desc"],
        "final_change": row["final_change"],
        "candidates": candidates,
        "target_image_ids": selected,
    }


class PositiveState:
    def __init__(self) -> None:
        self.rows = load_jsonl(INPUT)
        self.rows_by_id = {row["submission_id"]: row for row in self.rows}
        self.index_by_id = {row["submission_id"]: i for i, row in enumerate(self.rows)}
        if len(self.rows_by_id) != len(self.rows):
            raise ValueError(
                "positive_set_input.jsonl contains duplicate submission_id"
            )

        existing = load_jsonl(OUTPUT) if OUTPUT.exists() else []
        if len({row["submission_id"] for row in existing}) != len(existing):
            raise ValueError("positive_sets.jsonl contains duplicate submission_id")

        unknown = [
            row["submission_id"]
            for row in existing
            if row["submission_id"] not in self.rows_by_id
        ]
        if unknown:
            raise ValueError(
                "positive_sets.jsonl contains IDs outside positive_set_input.jsonl: "
                f"{unknown[0]}"
            )

        self.positives = {
            row["submission_id"]: validate_submission(
                self.rows_by_id[row["submission_id"]], row
            )
            for row in existing
        }
        self.boxes_by_image = load_box_index()
        self.lock = threading.Lock()

    def _meta(self, index: int, row: dict) -> dict:
        return {
            "index": index,
            "submission_id": row["submission_id"],
            "case_type": row["case_type"],
            "completed": row["submission_id"] in self.positives,
        }

    def tasks(self) -> dict:
        """Return lightweight task metadata for navigation and filtering."""

        tasks = [self._meta(index, row) for index, row in enumerate(self.rows)]
        completed = sum(item["completed"] for item in tasks)
        return {
            "total": len(tasks),
            "completed": completed,
            "pending": len(tasks) - completed,
            "tasks": tasks,
        }

    def task(self, index: int | None = None, submission_id: str | None = None) -> dict:
        """Load a task by stable submission ID, with index kept for navigation."""

        if submission_id is not None:
            row = self.rows_by_id.get(submission_id)
            if row is None:
                raise ValueError("unknown submission_id")
            index = self.index_by_id[submission_id]
        else:
            index = 0 if index is None else index
            if index < 0 or index >= len(self.rows):
                raise ValueError("task index out of range")
            row = self.rows[index]

        positive = self.positives.get(row["submission_id"])
        summary = self.tasks()
        return {
            "index": index,
            "total": summary["total"],
            "completed": summary["completed"],
            "pending": summary["pending"],
            "is_completed": positive is not None,
            "task": build_task_payload(row, self.boxes_by_image, positive),
        }

    def save(self, payload: dict) -> dict:
        submission_id = payload.get("submission_id")
        source = self.rows_by_id.get(submission_id)
        if source is None:
            raise ValueError("unknown submission_id")
        row = validate_submission(source, payload)

        with self.lock:
            self.positives[submission_id] = row
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
            write_jsonl(
                temporary,
                [
                    self.positives[source_row["submission_id"]]
                    for source_row in self.rows
                    if source_row["submission_id"] in self.positives
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
                submission_id = params.get("id", [None])[0]
                index = None
                if submission_id is None:
                    index = int(params.get("index", ["0"])[0])
                self._json(self.state.task(index=index, submission_id=submission_id))
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
