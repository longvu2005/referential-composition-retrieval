# Human annotation UIs

This directory contains two repo-local, file-backed annotation interfaces:

- `review/` for rewrite and Subject review;
- `positives/` for Full Positive selection.

Neither interface requires Label Studio. Canonical data remains under
`dataset/data/work/`, and both UIs write canonical JSONL directly.

## Rewrite review

Prepare or refresh the cumulative review catalog:

```bash
python tools/dataset/prepare_handoffs.py review
```

Run:

```bash
python -m labelstudio.review.app
```

Open:

```text
http://127.0.0.1:8090
```

The UI writes completed samples to:

```text
dataset/data/work/review/reviewed.jsonl
```

The left navigator keeps `All`, `Pending`, and `Reviewed` tasks available. Reviewed
tasks remain reopenable. Subject boxes are fixed metadata: they can be selected for
Subject assignment but cannot be moved, resized, created, or deleted.

## Full Positive selection

Prepare or refresh the cumulative positive catalog after rewrite review:

```bash
python tools/dataset/prepare_handoffs.py positives
```

Run:

```bash
python -m labelstudio.positives.app
```

Open:

```text
http://127.0.0.1:8091
```

Each task shows a sticky Query and reviewed instruction beside a vertical list of
identity-compatible target candidates. Subject boxes are read-only visual metadata.
The seed target is first, permanently selected, and cannot be removed. For every
other candidate, only the selection rail on the right changes its Full Positive
state; clicking the image opens it for inspection without changing the label.

Use `Save` to persist the current task or `Save & Next Pending` to continue through
unfinished tasks. `Cmd/Ctrl + Enter` also saves and advances. Completed tasks remain
reopenable and editable.

The UI writes directly to:

```text
dataset/data/work/positives/positive_sets.jsonl
```

Each canonical record remains:

```json
{
  "submission_id": "sample_000001",
  "positive_image_ids": ["seed_image_id", "additional_positive_id"]
}
```

Positive image IDs are stored in canonical candidate order, so repeated saves are
deterministic.

## Incremental behavior

Both handoff inputs are cumulative task catalogs:

```text
dataset/data/work/review/review_input.jsonl
dataset/data/work/positives/positive_set_input.jsonl
```

Re-running `prepare_handoffs.py` refreshes existing task metadata by
`submission_id`, keeps stable task order, and appends new tasks. Completion state is
stored separately in `reviewed.jsonl` and `positive_sets.jsonl`. This allows old
completed tasks to remain visible without re-running model work or creating a
second annotation format.
