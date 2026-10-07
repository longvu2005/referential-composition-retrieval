# Human annotation UIs

This directory contains two repo-local, file-backed annotation interfaces:

- `review/` for rewrite and Subject review;
- `positives/` for Full Positive selection.

Neither interface requires Label Studio. Canonical data remains under
`dataset/data/work/`, and both UIs write canonical JSONL directly.

## Rewrite review

Prepare or refresh the cumulative review catalog:

```bash
python scripts/data/prepare_handoffs.py review
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
python scripts/data/prepare_handoffs.py positives
```

Add `--clip-rerank` to rank only newly reviewed tasks with CLIP. Existing tasks
are never reranked. With no new task, CLIP is not loaded. Stop this UI before
running the finalization command because that launcher validates and may
rewrite the saved positive decisions. For example,
`bash scripts/data/phase2_finalize.bash --version 0.2.0` builds a new export after
the checked-in version `0.1.0`; this command also requires a working local
image tree.

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
  "sample_id": "train__query_id__target_id",
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

Re-running review preparation refreshes the review catalog by `sample_id`.
Positive preparation preserves migrated candidate order and checks that reviewed
text and Subject identities still match existing tasks; inspect positive labels
before applying any changed review. Completion state is
stored separately in `reviewed.jsonl` and `positive_sets.jsonl`. This allows old
completed tasks to remain visible without re-running model work or creating a
second annotation format.
