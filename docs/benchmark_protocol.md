# RCR benchmark protocol

Training, retrieval and reporting use the evaluator in `src/rcr/evaluation/`.

## Query, gallery and positives

Inference uses the reference image, Subject descriptions, requested change and
public Subject slots. Ground truth identities, head annotations, positives and
case labels provide supervision or evaluation metadata.

Each query searches all registered images in its original PIPA split, excluding
the reference image and leftover images. Rankings contain every eligible image
exactly once. Gallery order resolves coarse ties; coarse order resolves fine
shortlist ties.

**ID positives** contain every required identity, regardless of the requested
change. **Full positives** are the reviewed images satisfying the complete
query, intersected with the eligible gallery. Full positives must also be
ID positives.

## Metrics and paper tables

Average precision uses the complete ranking and positive set. mAP averages AP
over queries. R@K measures whether a query retrieves at least one positive in
its first K results. CandidateRecall measures the fraction of Full positives
retained in the coarse shortlist.

| Table | Metrics |
| --- | --- |
| Overall | ID-mAP, ID-R@1/5/10, Full-mAP, Full-R@1/5/10 |
| By-case | ID-mAP, ID-R@1, Full-mAP, Full-R@1 |

Overall metrics weight queries equally. By-case metrics cover INDIVIDUAL, GROUP,
DUAL and RELATIONAL; absent cases are left blank.

## Split and leakage policy

Train annotations supervise learning and sampling. Validation selects checkpoints
and retrieval coefficients. Test provides the final evaluation with the selected
model and settings frozen.

Scheduled validation calibrates the configured coverage, context and fusion
coefficients. The default checkpoint score averages overall Full-mAP and
macro-case Full-mAP. Calibration keeps the trained matching, pooling and coarse
shortlist fixed. Offline comparison may also select among saved validation
shortlists. `selected.yaml` binds the winning settings to the checkpoint and
validation fingerprints.

## Reproducibility and saved runs

Use the same final export and ordered gallery when comparing experiments.
Record the source revision, effective configuration, seed, runtime, checkpoint
and cache identities. Dataset counts and version come from `manifest.json`.

Saved results include `rankings.pt`, `run.json` and `metrics.json`. Evaluation
checks dataset and split consistency; reporting verifies the saved metric and
run metadata binding. Re-evaluate after changing annotations or rankings.
Use separate output directories for separate experiments.
