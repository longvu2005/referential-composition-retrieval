"""Minimal local UI for RCR rewrite review.

The review state is identity-based: selecting one box assigns the same identity
in both images. Bounding-box geometry is metadata and is never editable.
"""

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

from rcr.common.io import image_relative_path as _image_relative_path
from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.cases import CASE_TYPES, EDITABLE_CASE_TYPES, ONE_SUBJECT_CASES
from rcr.dataset.review import normalize_review_assignment
from rcr.dataset.rewrite import validate_review_output

INPUT = Path("dataset/data/work/review/review_input.jsonl")
OUTPUT = Path("dataset/data/work/review/reviewed.jsonl")
IMAGE_ROOT = Path("dataset/data/raw/images")
HTML = Path(__file__).with_name("app.html")


def _candidate_boxes(
    row: dict,
    image_id: str,
) -> list[dict]:
    candidate_ids = row.get("candidate_identity_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError(f"{row['sample_id']}: candidate_identity_ids are required")

    by_identity: dict[str, dict] = {}
    side = "query" if image_id == row["query_image_id"] else "target"
    for box in row[f"{side}_boxes"]:
        identity_id = box["identity_id"]
        if identity_id not in candidate_ids:
            continue
        if identity_id in by_identity:
            raise ValueError(
                f"{row['sample_id']}: duplicate box for identity {identity_id} "
                f"in {image_id}"
            )
        by_identity[identity_id] = box

    missing = set(candidate_ids) - set(by_identity)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"{row['sample_id']}: missing candidate boxes in {image_id}: {missing_text}"
        )

    return [
        {
            "identity_id": identity_id,
            "x": 100 * by_identity[identity_id]["x"],
            "y": 100 * by_identity[identity_id]["y"],
            "width": 100 * by_identity[identity_id]["width"],
            "height": 100 * by_identity[identity_id]["height"],
        }
        for identity_id in candidate_ids
    ]


def _initial_assignment(row: dict, review: dict | None) -> tuple[str, list[dict]]:
    if review is not None:
        return normalize_review_assignment(
            row["sample_id"],
            review["case_type"],
            review["subjects"],
            row["candidate_identity_ids"],
        )
    return row["case_type"], [
        {
            "subject_id": subject["subject_id"],
            "identity_ids": list(subject["identity_ids"]),
        }
        for subject in row["subjects"]
    ]


def build_task_payload(
    row: dict,
    review: dict | None = None,
) -> dict:
    """Build one browser task from canonical normalized head boxes."""

    case_type, subjects = _initial_assignment(row, review)
    final_desc = review["final_desc"] if review is not None else row.get("final_desc")
    final_change = (
        review["final_change"] if review is not None else row.get("final_change")
    )

    return {
        "sample_id": row["sample_id"],
        "case_type": case_type,
        "subjects": subjects,
        "candidate_identity_ids": row["candidate_identity_ids"],
        "query": {
            "image_id": row["query_image_id"],
            "image_url": (
                "/api/image?path="
                f"{_image_relative_path(row['query_image_url']).as_posix()}"
            ),
            "boxes": _candidate_boxes(row, row["query_image_id"]),
        },
        "target": {
            "image_id": row["target_image_id"],
            "image_url": (
                "/api/image?path="
                f"{_image_relative_path(row['target_image_url']).as_posix()}"
            ),
            "boxes": _candidate_boxes(row, row["target_image_id"]),
        },
        "final_desc": final_desc or "",
        "final_change": final_change or "",
        "original": {
            "case_type": row["case_type"],
            "subjects": row["subjects"],
            "final_desc": row.get("final_desc"),
            "final_change": row.get("final_change"),
        },
    }


def validate_submission(source: dict, payload: dict) -> dict:
    """Validate one browser submission and return the canonical review row."""

    sample_id = source["sample_id"]
    if payload.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id}: sample_id mismatch")

    case_type, subjects = normalize_review_assignment(
        sample_id,
        payload.get("case_type"),
        payload.get("subjects"),
        source["candidate_identity_ids"],
    )
    final_desc, final_change = validate_review_output(
        case_type,
        {
            "final_desc": payload.get("final_desc"),
            "final_change": payload.get("final_change"),
        },
    )
    return {
        "sample_id": sample_id,
        "case_type": case_type,
        "subjects": subjects,
        "final_desc": final_desc,
        "final_change": final_change,
    }


class ReviewState:
    def __init__(self) -> None:
        self.rows = load_jsonl(INPUT)
        self.rows_by_id = {row["sample_id"]: row for row in self.rows}
        self.index_by_id = {row["sample_id"]: i for i, row in enumerate(self.rows)}
        if len(self.rows_by_id) != len(self.rows):
            raise ValueError("review_input.jsonl contains duplicate sample_id")

        existing = load_jsonl(OUTPUT) if OUTPUT.exists() else []
        self.reviews = {row["sample_id"]: row for row in existing}
        if len(self.reviews) != len(existing):
            raise ValueError("reviewed.jsonl contains duplicate sample_id")
        self.lock = threading.Lock()

    def _meta(self, index: int, row: dict) -> dict:
        review = self.reviews.get(row["sample_id"])
        return {
            "index": index,
            "sample_id": row["sample_id"],
            "case_type": (
                review["case_type"] if review is not None else row["case_type"]
            ),
            "reviewed": review is not None,
        }

    def tasks(self) -> dict:
        """Return lightweight task metadata for navigation and filtering."""

        tasks = [self._meta(index, row) for index, row in enumerate(self.rows)]
        completed = sum(item["reviewed"] for item in tasks)
        return {
            "total": len(tasks),
            "completed": completed,
            "pending": len(tasks) - completed,
            "case_types": list(CASE_TYPES),
            "editable_case_types": list(EDITABLE_CASE_TYPES),
            "one_subject_case_types": sorted(ONE_SUBJECT_CASES),
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

        review = self.reviews.get(row["sample_id"])
        summary = self.tasks()
        return {
            "index": index,
            "total": summary["total"],
            "completed": summary["completed"],
            "pending": summary["pending"],
            "reviewed": review is not None,
            "task": build_task_payload(row, review),
        }

    def save(self, payload: dict) -> dict:
        sample_id = payload.get("sample_id")
        source = self.rows_by_id.get(sample_id)
        if source is None:
            raise ValueError("unknown sample_id")
        row = validate_submission(source, payload)

        with self.lock:
            self.reviews[sample_id] = row
            temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
            # Dict insertion order preserves saved order when a review is edited.
            write_jsonl(temporary, self.reviews.values())
            os.replace(temporary, OUTPUT)
        return row


class ReviewHandler(BaseHTTPRequestHandler):
    state: ReviewState

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
        except Exception as error:  # local review server: surface actionable errors
            self._error(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            row = self.state.save(payload)
            self._json({"review": row, "summary": self.state.tasks()})
        except Exception as error:  # local review server: surface actionable errors
            self._error(error)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RCR rewrite-review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    state = ReviewState()
    handler = type("BoundReviewHandler", (ReviewHandler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"RCR rewrite review: http://{args.host}:{args.port}")
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
