# Full Positive audit

The audit checks annotation integrity, identity compatibility and review coverage
within each query's gallery. It produces diagnostic reports and optional manual
review tasks.

## Run the audit

The labels-only audit requires the dataset environment:

```bash
.venv-dataset/bin/python tools/data/audit.py \
  --final-dir dataset/data/final --output runs/data_audit.json \
  --review-queue runs/full_positive_review.jsonl
```

Supply `--positive-catalog` and `--positive-decisions` to check the original
review evidence. `--strict` returns a failing status when unresolved findings
remain, after saving the report.

## Interpret findings

The report covers invalid annotations, query conflicts, out-of-split labels
and review scope.

| Scope | Meaning |
| --- | --- |
| `unknown` | Required review evidence is absent |
| `stale` | Review evidence differs from current data |
| `partial_catalog` | The catalog covers part of the eligible candidate set |
| `complete_catalog` | Matching review evidence covers all eligible candidates |

Review coverage does not establish semantic correctness. ID-compatible images
outside the Full Positive set still require human judgment.

## Manual review

Open the generated queue with separate review outputs:

```bash
.venv-dataset/bin/python -m labelstudio.positives.app \
  --input runs/full_positive_review.jsonl \
  --output runs/full_positive_reviewed.jsonl
```

Reconcile accepted decisions with canonical work files before rebuilding the
dataset. The audit leaves canonical labels unchanged. New annotations require
fresh evaluation and validation selection under the
[benchmark protocol](benchmark_protocol.md).

Optional `--detection-cache` adds GT-to-detection coverage diagnostics using the
independent detection cache.
