# Human review handoffs

This directory contains two independent interfaces: the repo-local Rewrite Review
UI under `review/` and the offline Label Studio handoff for Full Positive selection
under `positives/`. Rewrite Review does not use Label Studio. Canonical data remains
under `dataset/data/work/`, and the repo never calls the Label Studio SDK or API.

## Rewrite review

Prepare the canonical review input if needed:

```bash
python tools/dataset/prepare_handoffs.py review
```

Run the repo-local review UI:

```bash
python -m labelstudio.review.app
```

Open:

```text
http://127.0.0.1:8090
```

The UI writes completed samples directly to:

```text
dataset/data/work/review/reviewed.jsonl
```

The review UI is a task workspace rather than a strictly linear queue. The left
navigator shows all tasks and supports `All`, `Pending`, and `Reviewed` status
filters, case-type filtering, and search by task number or `submission_id`. Reviewed
tasks remain reopenable. `Save & Next Pending` skips completed tasks, while `Prev`
and `Next` still allow sequential inspection. Unsaved edits are protected before
navigating away.

Within each task:

1. choose `SINGLE`, `MULTI`, or `RELATIONAL`;
2. assign identities by clicking fixed person boxes;
3. review `final_desc` and `final_change`.

`SINGLE` has only `S1`; `MULTI` and `RELATIONAL` require both `S1` and `S2`.
The same identity box is linked across Query and Target: clicking either side
updates both images immediately. Boxes are metadata, not annotations, so they
cannot be moved, resized, created, or deleted. Zoom and pan transform the image
and boxes together. Selected boxes display only `S1` or `S2`; identity IDs remain
internal.

The assignment state is single-valued (`identity -> Subject`), so one identity
cannot belong to two Subjects. Clicking an identity already assigned to the other
Subject reassigns it; clicking an identity already assigned to the active Subject
unassigns it. Server-side validation enforces the same invariant before writing
canonical output.

The candidate set remains restricted to identities present in both Query and the
seed Target. Changing the reviewed case also changes the canonical Subject
structure: `SINGLE` stores only Subject 1, while the other cases store Subjects 1
and 2.

## Label Studio setup for Full Positive selection

Install and start Label Studio separately only for the Full Positive handoff:

```bash
python -m pip install label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/dataset/data/raw/images"
label-studio start
```

Create `RCR Positive Selection` with `labelstudio/positives/config.xml` and enable
**Allow empty annotations** so a group may contain zero valid candidates.

## Full Positive selection

Prepare the canonical candidate input:

```bash
python tools/dataset/prepare_handoffs.py positives
```

Create the Label Studio task file:

```bash
python -m labelstudio.positives.prepare
```

Import this file manually in the `RCR Positive Selection` project:

```text
dataset/data/work/positives/labelstudio_tasks.json
```

Each task contains at most 10 non-seed candidates from one sample. The seed target
is shown only as reference and is always retained as a positive.

After annotation, export the project as JSON and save it as:

```text
dataset/data/work/positives/labelstudio_export.json
```

Collect completed groups:

```bash
python -m labelstudio.positives.collect
```

Canonical output:

```text
dataset/data/work/positives/positive_sets.jsonl
```

A sample is collected only after every candidate group for that sample has a
completed annotation. Existing canonical outputs are merged by `submission_id`.

## Incremental behavior

The incremental boundary remains the canonical handoff files:

- `prepare_handoffs.py review` maintains `review_input.jsonl` as a cumulative task
  catalog, refreshing known IDs and appending new ones without dropping reviewed
  tasks.
- `reviewed.jsonl` is the canonical rewrite-review completion state used by the UI.
- `prepare_handoffs.py positives` excludes IDs already in `positive_sets.jsonl`.
- the rewrite UI merges each completed sample directly into `reviewed.jsonl`;
- Positive `prepare.py` and `collect.py` keep the offline Label Studio handoff deterministic.

Therefore completed samples are not prepared again when new data is added.
