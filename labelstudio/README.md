# Annotation UIs

Two local, file-backed interfaces support text/Subject review and Full Positive
selection. Both use canonical JSONL files under `dataset/data/work/`.
Install the dataset environment with `bash scripts/setup.bash dataset`.

## Text and Subject review

Prepare tasks and launch the interface:

```bash
bash scripts/data/prepare_review.bash
.venv-dataset/bin/python -m labelstudio.review.app
```

Open <http://127.0.0.1:8090>. Review the text and Subject assignments, then save
completed tasks. Subject boxes are fixed metadata. Saved tasks remain available
for later edits.

## Full Positive selection

After text review, prepare candidates and launch the interface:

```bash
bash scripts/data/prepare_positives.bash
.venv-dataset/bin/python -m labelstudio.positives.app
```

Open <http://127.0.0.1:8091>. Select images that satisfy the complete query.
Candidates are identity-compatible images in the query's split. The original
seed target stays selected. Save the task to persist decisions; completed tasks
can be reopened.

Optional `--clip-rerank` during preparation orders new tasks using CLIP. See the
[dataset guide](../dataset/README.md) for environment setup and finalization.

## Saved data

| Interface | Task catalog | Decisions |
| --- | --- | --- |
| Review | `review/review_input.jsonl` | `review/reviewed.jsonl` |
| Positives | `positives/positive_set_input.jsonl` | `positives/positive_sets.jsonl` |

Paths are relative to `dataset/data/work/`. Preparation refreshes task catalogs
while retaining completed decisions. Review edits may require rechecking affected
positive labels. Stop the relevant UI before refreshing or finalizing files.

For a separate review queue, the positive UI accepts `--input`, `--output` and
`--image-root`. Keep audit outputs separate until reviewed decisions are
reconciled with canonical data. See the [audit guide](../docs/full_positive_audit.md).
