"""Run chunked Gemini Batch rewriting for dataset annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from rcr.dataset.gemini_batch import (
    build_batch_request,
    chunk_records,
    merge_rewrite_outputs,
    parse_model_output,
    select_pending_records,
)
from rcr.dataset.rewrite import build_rewrite_result
from rcr.utils.jsonl import load_jsonl, write_jsonl

MODEL = "gemini-3.6-flash"
DEFAULT_CHUNK_SIZE = 1000

PROMPT = Path("dataset/prompts/rewrite.txt")

PRODUCTION_INPUT = Path("dataset/data/work/rewrite/rewrite_input.jsonl")
PRODUCTION_OUTPUT = Path("dataset/data/work/rewrite/rewrite_output.jsonl")
PRODUCTION_WORK_DIR = Path("dataset/data/work/rewrite")

STRESS_INPUT = Path("tests/fixtures/rewrite_prompt_stress.jsonl")
STRESS_OUTPUT = Path("dataset/data/work/rewrite/prompt_test/rewrite_output.jsonl")
STRESS_WORK_DIR = Path("dataset/data/work/rewrite/prompt_test")

LOCAL_READY = "LOCAL_READY"
TERMINAL_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
CHUNK_RE = re.compile(r"chunk_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    split.add_argument("--limit", type=int)
    split.add_argument(
        "--case-type",
        choices=["SINGLE", "MULTI", "RELATIONAL"],
    )
    split.add_argument("--stress-test", action="store_true")

    submit = subparsers.add_parser("submit")
    submit.add_argument("--chunk")
    submit.add_argument("--stress-test", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--chunk")
    status.add_argument("--stress-test", action="store_true")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--chunk")
    collect.add_argument("--stress-test", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--stress-test", action="store_true")

    return parser.parse_args()


def get_paths(stress_test: bool) -> tuple[Path, Path, Path]:
    if stress_test:
        return STRESS_INPUT, STRESS_OUTPUT, STRESS_WORK_DIR
    return PRODUCTION_INPUT, PRODUCTION_OUTPUT, PRODUCTION_WORK_DIR


def make_client():
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=5,
                initial_delay=2,
                max_delay=30,
            )
        ),
    )


def now() -> str:
    return datetime.now(UTC).isoformat()


def chunk_root(work_dir: Path) -> Path:
    return work_dir / "chunks"


def manifest_paths(work_dir: Path) -> list[Path]:
    return sorted(chunk_root(work_dir).glob("chunk_*/manifest.json"))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def next_chunk_number(work_dir: Path) -> int:
    numbers = []
    for path in manifest_paths(work_dir):
        match = CHUNK_RE.fullmatch(path.parent.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def empty_rewrite(submission_id: str) -> dict:
    """Represent a processed sample that needs manual rewrite review."""

    return {
        "submission_id": submission_id,
        "final_desc": None,
        "final_change": None,
    }



def completed_ids(output_path: Path, work_dir: Path) -> set[str]:
    """Return every sample already processed by Gemini at sample level."""

    ids = set()

    if output_path.exists():
        ids.update(row["submission_id"] for row in load_jsonl(output_path))

    for path in manifest_paths(work_dir):
        for name in ("output.jsonl", "errors.jsonl"):
            chunk_path = path.parent / name
            if chunk_path.exists():
                ids.update(row["submission_id"] for row in load_jsonl(chunk_path))

    return ids


def collected_chunk_outputs(work_dir: Path) -> list[dict]:
    """Load chunk outputs, converting legacy sample errors to empty rewrites."""

    outputs = {}

    for path in manifest_paths(work_dir):
        chunk_output = path.parent / "output.jsonl"
        if chunk_output.exists():
            for row in load_jsonl(chunk_output):
                outputs[row["submission_id"]] = row

    for path in manifest_paths(work_dir):
        errors_path = path.parent / "errors.jsonl"
        if not errors_path.exists():
            continue

        for error in load_jsonl(errors_path):
            submission_id = error["submission_id"]
            outputs.setdefault(submission_id, empty_rewrite(submission_id))

    return list(outputs.values())


def process_batch_results(
    source_records: list[dict],
    raw_bytes: bytes,
) -> tuple[list[dict], list[dict]]:
    """Convert every sample-level result into a rewrite or manual-review blank."""

    source_by_id = {row["submission_id"]: row for row in source_records}
    outputs = {}
    errors = {}

    for line in raw_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue

        result = json.loads(line)
        submission_id = result.get("key")

        if submission_id not in source_by_id:
            raise ValueError(f"Unknown batch result key: {submission_id!r}")
        if submission_id in outputs:
            raise ValueError(f"Duplicate batch result key: {submission_id}")

        try:
            model_output = parse_model_output(result)
            outputs[submission_id] = build_rewrite_result(
                source_by_id[submission_id],
                model_output,
            )
        except Exception as exc:
            outputs[submission_id] = empty_rewrite(submission_id)
            errors[submission_id] = {
                "submission_id": submission_id,
                "error": str(exc),
            }

    for source in source_records:
        submission_id = source["submission_id"]
        if submission_id in outputs:
            continue

        outputs[submission_id] = empty_rewrite(submission_id)
        errors[submission_id] = {
            "submission_id": submission_id,
            "error": "Batch result is missing for this sample.",
        }

    ordered_outputs = [outputs[row["submission_id"]] for row in source_records]
    ordered_errors = [
        errors[row["submission_id"]]
        for row in source_records
        if row["submission_id"] in errors
    ]
    return ordered_outputs, ordered_errors


def assigned_ids(work_dir: Path) -> set[str]:
    ids = set()

    for path in manifest_paths(work_dir):
        manifest = load_manifest(path)
        state = manifest["state"]

        if manifest.get("collected_at") is not None:
            continue
        if state in TERMINAL_STATES:
            continue

        ids.update(manifest["submission_ids"])

    return ids


def active_remote_manifests(work_dir: Path) -> list[Path]:
    active = []

    for path in manifest_paths(work_dir):
        manifest = load_manifest(path)
        if manifest.get("batch_name") is None:
            continue
        if manifest.get("collected_at") is not None:
            continue
        if manifest["state"] in TERMINAL_STATES:
            continue
        active.append(path)

    return active


def ready_manifest_paths(work_dir: Path) -> list[Path]:
    return [
        path
        for path in manifest_paths(work_dir)
        if load_manifest(path)["state"] == LOCAL_READY
        and load_manifest(path).get("batch_name") is None
    ]


def select_manifest(
    work_dir: Path,
    chunk: str | None,
    *,
    require_remote: bool,
) -> Path:
    if chunk is not None:
        chunk_id = chunk if chunk.startswith("chunk_") else f"chunk_{int(chunk):06d}"
        path = chunk_root(work_dir) / chunk_id / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown chunk: {chunk}")
        manifest = load_manifest(path)
        if require_remote and manifest.get("batch_name") is None:
            raise RuntimeError(f"Chunk is not submitted: {chunk_id}")
        return path

    if require_remote:
        paths = active_remote_manifests(work_dir)
        if not paths:
            raise FileNotFoundError("No submitted chunk is waiting for status/collect.")
        if len(paths) > 1:
            raise RuntimeError("Multiple remote chunks found; select one with --chunk.")
        return paths[0]

    paths = ready_manifest_paths(work_dir)
    if not paths:
        raise FileNotFoundError("No ready chunk. Run phase1_rewrite.bash prepare.")
    return paths[0]


def split(args: argparse.Namespace) -> None:
    input_path, output_path, work_dir = get_paths(args.stress_test)
    records = load_jsonl(input_path)
    prompt = PROMPT.read_text(encoding="utf-8").strip()

    pending = select_pending_records(
        records,
        completed_ids=completed_ids(output_path, work_dir),
        in_flight_ids=assigned_ids(work_dir),
        case_type=args.case_type,
        limit=args.limit,
    )

    if not pending:
        print("No new samples to split.")
        return

    start = next_chunk_number(work_dir)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    chunks = chunk_records(pending, args.chunk_size)

    for offset, records_chunk in enumerate(chunks):
        chunk_id = f"chunk_{start + offset:06d}"
        chunk_dir = chunk_root(work_dir) / chunk_id
        input_chunk_path = chunk_dir / "input.jsonl"
        requests_path = chunk_dir / "requests.jsonl"

        write_jsonl(input_chunk_path, records_chunk)
        write_jsonl(
            requests_path,
            (build_batch_request(prompt, record) for record in records_chunk),
        )

        save_manifest(
            chunk_dir / "manifest.json",
            {
                "chunk_id": chunk_id,
                "state": LOCAL_READY,
                "model": MODEL,
                "prompt_sha256": prompt_hash,
                "num_requests": len(records_chunk),
                "submission_ids": [row["submission_id"] for row in records_chunk],
                "uploaded_file": None,
                "batch_name": None,
                "created_at": now(),
                "submitted_at": None,
                "collected_at": None,
            },
        )

    print(f"Pending: {len(pending)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk size: {args.chunk_size}")


def submit(args: argparse.Namespace) -> None:
    from google.genai import types

    _, _, work_dir = get_paths(args.stress_test)

    active = active_remote_manifests(work_dir)
    if active:
        manifest = load_manifest(active[0])
        raise RuntimeError(
            f"Chunk {manifest['chunk_id']} is still active: {manifest['state']}. "
            "Run status/collect before submitting the next chunk."
        )

    manifest_path = select_manifest(
        work_dir,
        args.chunk,
        require_remote=False,
    )
    manifest = load_manifest(manifest_path)
    chunk_dir = manifest_path.parent
    requests_path = chunk_dir / "requests.jsonl"

    with make_client() as client:
        uploaded_name = manifest.get("uploaded_file")
        if uploaded_name is None:
            uploaded = client.files.upload(
                file=str(requests_path),
                config=types.UploadFileConfig(
                    display_name=f"rcr-rewrite-{manifest['chunk_id']}",
                    mime_type="jsonl",
                ),
            )
            uploaded_name = uploaded.name
            manifest["uploaded_file"] = uploaded_name
            save_manifest(manifest_path, manifest)

        job = client.batches.create(
            model=manifest["model"],
            src=uploaded_name,
            config={"display_name": f"rcr-rewrite-{manifest['chunk_id']}"},
        )

    manifest["batch_name"] = job.name
    manifest["state"] = job.state.name
    manifest["submitted_at"] = now()
    save_manifest(manifest_path, manifest)

    print(f"Chunk: {manifest['chunk_id']}")
    print(f"Submitted: {manifest['num_requests']}")
    print(f"Gemini batch: {manifest['batch_name']}")
    print(f"State: {manifest['state']}")


def status(args: argparse.Namespace) -> None:
    _, _, work_dir = get_paths(args.stress_test)
    path = select_manifest(work_dir, args.chunk, require_remote=True)
    manifest = load_manifest(path)

    with make_client() as client:
        job = client.batches.get(name=manifest["batch_name"])

    manifest["state"] = job.state.name
    save_manifest(path, manifest)

    print(f"Chunk: {manifest['chunk_id']}")
    print(f"Gemini batch: {manifest['batch_name']}")
    print(f"State: {manifest['state']}")
    print(f"Requests: {manifest['num_requests']}")


def collect(args: argparse.Namespace) -> None:
    _, _, work_dir = get_paths(args.stress_test)
    manifest_path = select_manifest(work_dir, args.chunk, require_remote=True)
    manifest = load_manifest(manifest_path)

    with make_client() as client:
        job = client.batches.get(name=manifest["batch_name"])
        manifest["state"] = job.state.name
        save_manifest(manifest_path, manifest)

        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Chunk is not ready: {job.state.name}")

        raw_bytes = client.files.download(file=job.dest.file_name)

    chunk_dir = manifest_path.parent
    (chunk_dir / "results.jsonl").write_bytes(raw_bytes)

    source_records = load_jsonl(chunk_dir / "input.jsonl")
    outputs, errors = process_batch_results(source_records, raw_bytes)

    write_jsonl(chunk_dir / "output.jsonl", outputs)
    write_jsonl(chunk_dir / "errors.jsonl", errors)

    manifest["state"] = "JOB_STATE_SUCCEEDED"
    manifest["collected_at"] = now()
    manifest["num_succeeded"] = len(outputs) - len(errors)
    manifest["num_failed"] = len(errors)
    save_manifest(manifest_path, manifest)

    print(f"Chunk: {manifest['chunk_id']}")
    print(f"Collected: {len(outputs)}")
    print(f"Failed: {len(errors)}")


def merge(args: argparse.Namespace) -> None:
    input_path, output_path, work_dir = get_paths(args.stress_test)
    records = load_jsonl(input_path)
    existing = load_jsonl(output_path) if output_path.exists() else []

    additions = collected_chunk_outputs(work_dir)
    outputs = merge_rewrite_outputs(records, existing, additions)
    output_ids = {row["submission_id"] for row in outputs}
    missing = [
        row["submission_id"]
        for row in records
        if row["submission_id"] not in output_ids
    ]

    if missing:
        raise RuntimeError(
            f"Cannot merge final rewrite output: {len(missing)} samples are missing. "
            "Collect remaining chunks or retry a failed batch job."
        )

    write_jsonl(output_path, outputs)
    print(f"Merged: {len(outputs)}")
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()

    if args.action == "split":
        split(args)
    elif args.action == "submit":
        submit(args)
    elif args.action == "status":
        status(args)
    elif args.action == "collect":
        collect(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
