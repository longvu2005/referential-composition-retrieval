# Frozen FAFA with matched and Subject means

`method: fafa_pooling` is an inference-only FAFA adapter variant. It emits both
`matched_mean` and `subject_mean`, without fitting weights or selecting a winner.
The original `method: fafa`, its configuration and its component-mean score remain
available unchanged. This is an RCR adapter experiment, not official FAFA scoring.

## Shared inputs and assignment

Use the same released checkpoint, shared DINO detections, CLIP reference selector,
predicted member sets, condition-text routing and native FDA features as FAFA.
Only public Subject slot IDs enter aggregation. Case labels, GT identity IDs,
GT member counts and positives never enter inference.

For each query/gallery image, rectangular max-sum Hungarian produces the same
one-to-one assignment for both poolings. Real pairs remain mandatory, even at low
FDA similarity. There is no learned identity head, private null, new threshold,
background logit, context score, coarse shortlist or RCR training in this method.
In particular, proposed's background logit -3.5 is not FAFA's unmatched score.

## Pooling definitions

Let `n_s` be the number of predicted reference components in Subject `s`,
`m_s` its number of matched components, and `a_s` their summed native FDA score.
All components in a Subject receive uniform weight `1/n_s`; these are not
proposed's learned grounding confidences. With the existing FAFA unmatched score
`b = -1`:

- `matched_mean = sum_s(a_s/n_s) / sum_s(m_s/n_s)`, ignoring terms with `n_s=0`.
  If no component is matched, return `b`.
- `subject_mean = mean_s(a_s/m_s if m_s>0 else b)`, retaining every public Subject.

These are the same two aggregation rules as proposed, with uniform predicted
membership weights and native FDA matching instead of learned identity matching.
The original FAFA score averages over every reference component and penalizes
each unmatched component. The new rules renormalize partial matches within a
Subject. `matched_mean` ignores empty roles; `subject_mean` retains their penalty.
All new scores use the same assignment, without optimizing assignment separately
for the final pooling. `K<=2` reuses the original vectorized assignment shortcuts.

## Run

Use the existing FAFA environment and prepared shared detection cache:

```bash
bash scripts/setup.bash fafa
bash scripts/methods/fafa_pooling.bash val
```

Or, when FAFA assets are already prepared:

```bash
.venv-fafa/bin/python tools/run.py run --config configs/fafa_pooling.yaml --split val
```

Both outputs have `scores.npy`, `rankings.pt`, `run.json` and official `metrics.json`:

```text
runs/fafa_pooling/matched_mean/val/
runs/fafa_pooling/subject_mean/val/
```

`runs/fafa_pooling/summary.csv` compares the two outputs from the latest invocation.
`retrieve` writes both rankings; `evaluate` evaluates both saved rankings. `train`,
`build-cache` and `prepare-supervision` reject this method. Its setup uses
`.venv-fafa`; it does not need a separate environment.

Feature and component-score cache fingerprints are unchanged, so valid FAFA
caches under `cache/fafa/features` are reused. Assignment and aggregation are
computed again. Final image scores or rankings alone do not contain sufficient
component evidence. Keep different configurations in separate output roots.

Complete validation freezes both variants in this method's own `protocol.json`.
Test checks that lock and evaluates both predetermined variants, without selecting
from test metrics:

```bash
bash scripts/methods/fafa_pooling.bash test
.venv-fafa/bin/python tools/report.py --split val \
  --methods fafa fafa_matched_mean fafa_subject_mean
```

The report's default method list remains unchanged; request the two new rows
explicitly. Compare the original FAFA adapter and both variants on the same
benchmark fingerprint before attributing improvements to proposed's learned parts.
