# Label Studio handoffs

This directory contains only the offline files and converters needed for the two
human-labeling handoffs. Canonical data remains under `dataset/data/work/`.
There is no Label Studio SDK, API key, project ID, push, or pull step in the repo.

## Setup

Install and start Label Studio separately if needed:

```bash
python -m pip install label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/dataset/data/raw/images"
label-studio start
```

Create two projects manually:

- `RCR Rewrite Review`: use `labelstudio/review/config.xml`.
- `RCR Positive Selection`: use `labelstudio/positives/config.xml` and enable
  **Allow empty annotations** so a group may contain zero valid candidates.

## Rewrite review

Prepare the canonical handoff input if needed:

```bash
python tools/dataset/prepare_handoffs.py review
```

Create the Label Studio task file:

```bash
python -m labelstudio.review.prepare
```

Import this file manually in the `RCR Rewrite Review` project:

```text
dataset/data/work/review/labelstudio_tasks.json
```

After annotation, export the project as JSON and save it as:

```text
dataset/data/work/review/labelstudio_export.json
```

Collect completed annotations:

```bash
python -m labelstudio.review.collect
```

Canonical output:

```text
dataset/data/work/review/reviewed.jsonl
```

The UI shows Query and Target with Subject-colored boxes, Subject identity IDs,
a required Case Type dropdown preselected to the existing case, and editable
`final_desc` / `final_change`. Reviewers may correct the case type. Empty Gemini
rewrites remain empty required text boxes for human correction. New reviewed
records store `case_type`; legacy reviewed records without it still fall back to
the original annotation case.

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

- `prepare_handoffs.py review` excludes IDs already in `reviewed.jsonl`.
- `prepare_handoffs.py positives` excludes IDs already in `positive_sets.jsonl`.
- `prepare.py` deterministically rewrites the current offline import file.
- `collect.py` merges completed exported annotations into the cumulative output.

Therefore completed samples are not prepared again when new data is added.
