# Full Positive audit and manual review

This audit checks annotation integrity and the scope of the original review
catalog against **all** ID-compatible images in each query's split. It never
runs an encoder, changes labels, changes negative sampling, or infers that an
ID-compatible image must satisfy the requested condition.

## Checked-in snapshot

Audit of version `0.1.0`, base commit `4234335804a450fe7aa62d5637798844de357ad4`
(20 October 2026), using the checked-in annotations without a work catalog:

| Split | Queries | ID Positive pairs | Full Positive pairs | Invalid queries | Conflict groups / queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4,033 | 362,558 | 9,892 | 0 | 6 / 12 |
| val | 264 | 21,653 | 757 | 0 | 1 / 2 |
| test | 14 | 714 | 18 | 0 | 0 / 0 |

Pairs count query/image relationships, not unique gallery images. Full Positive
counts are after split projection. The stored lists also contain 1,387/370/2
positive pairs outside the train/val/test galleries respectively; these are
reported and excluded in memory under the existing protocol.

All 4,311 queries have **unknown review scope** here because the original
`positive_set_input.jsonl` and `positive_sets.jsonl` were not supplied. The
352,666/20,896/696 ID-compatible non-Full pairs are candidates whose rejection
scope cannot be established from this export alone. They are **not measured
missing positives**, and unknown scope is not evidence of incorrect annotation.

Equivalent-query conflicts mean the same case label, exact condition and ordered
Subject identities disagree on some non-self gallery images. There are 19 disputed
negative pairs on train, already excluded by the existing training sampler.
No positive sets are merged. Review the JSON conflict records before deciding
which annotation is correct. An empty conflict list does not certify completeness.

## Run the audit

A labels-only run uses the dataset environment and needs no images or model
weights. Output JSON records hashes of the final export and supplied review files.

```bash
.venv-dataset/bin/python tools/data/audit.py \
  --final-dir dataset/data/final \
  --output runs/data_audit.json \
  --review-queue runs/full_positive_review.jsonl
```

To establish review-catalog scope, supply both original work files:

```bash
.venv-dataset/bin/python tools/data/audit.py \
  --positive-catalog dataset/data/work/positives/positive_set_input.jsonl \
  --positive-decisions dataset/data/work/positives/positive_sets.jsonl \
  --output runs/data_audit_with_scope.json \
  --review-queue runs/full_positive_review.jsonl \
  --strict
```

`--strict` saves the report first, then exits 1 if annotation errors, conflicts,
stale review evidence or incomplete/unknown review scope remain. Without it,
findings are reported without turning a diagnostic into a pipeline gate.

| `scope_status` | Meaning |
| --- | --- |
| `unknown` | Catalog or submitted decision is absent for this query |
| `stale` | Text/Subjects/query/seed changed, a decision is invalid, or its split positive set differs from final |
| `partial_catalog` | Matching submitted review exists, but some split ID Positives were absent from its catalog |
| `complete_catalog` | Matching submitted review covers every split ID Positive |

Candidates outside the query split do not contribute to scope or retrieval.
Catalog ordering does not affect coverage. Per-query output lists ID Positives,
Full Positives, out-of-split labels, invalid identities, unconfirmed negatives,
catalog negatives and conflicts. `catalog_negative_image_ids` means rejection
in a matching submitted decision, not independently verified semantic truth.
Even `complete_catalog` cannot establish that every candidate was viewed or
labeled correctly: semantic completeness remains a human-review question.

## Review unresolved queries separately

The queue contains structurally valid unresolved/conflicting queries. It includes
all split ID Positives, with the seed first and the current split positive set
prefilled. Structurally invalid queries stay in the report and must be repaired
before using the positive UI. Audit queue generation does not mark any task done.

```bash
.venv-dataset/bin/python -m labelstudio.positives.app \
  --input runs/full_positive_review.jsonl \
  --output runs/full_positive_reviewed.jsonl \
  --image-root dataset/data/raw/images
```

Use a fresh review-output path to inspect unresolved tasks. Decisions are saved
only through the UI. Reconcile reviewed decisions with the canonical work files
and build a new dataset version deliberately; this audit does not write back or
merge them. New labels require fresh evaluation/validation selection under the
existing fingerprint checks. Do not tune a model using final-test review results.

## Detection diagnostics after cache separation

Optional detection coverage aligns GT heads to predicted boxes during the audit,
never from cached identity labels:

```bash
.venv-proposed/bin/python tools/data/audit.py \
  --detection-cache cache/grounding-dino-tiny \
  --image-root dataset/data/raw/images \
  --output runs/data_and_detection_audit.json
```

Detection diagnostics need no scene or FAFA features. Historical unified scene
caches remain supported through `--cache`, with `--scene-cache` for a historical
FAFA wrapper. Legacy DINO sources lacking pixel boxes and scene-size metadata need
`--scene-size HEIGHT WIDTH` with the original letterbox size and accessible raw
images. Coverage measures GT-to-detection alignment, not localization AP. GT for
val/test is permitted in this diagnostic, separately from training/scoring.
