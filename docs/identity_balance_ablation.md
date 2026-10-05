# Identity balance ablations A–D

This suite isolates identity amplitude at the two existing token additions. It
reuses the visual cache, model, loss, train loop, retrieval path, evaluator, and
`tools/methods/run.py`. There is no second model or training implementation.

## Design and code ownership

| Location | Responsibility |
| --- | --- |
| `configs/methods/proposed.yaml` | Main settings: both fusion coefficients are explicitly 0.4; normal identity behavior is `mode: none`. |
| `configs/ablations/identity_balance.yaml` | Eight A–D configurations, shared protocol, reference checkpoint, selection and extra seeds. |
| `identity_balance.py` | Parameter-free identity transform, initial train calibration, initialization fingerprint and masked norm moments. |
| `composition.py` / `reasoning.py` | Apply that transform at the existing query / target identity projections. |
| `shortlist.py` | Validate the frozen coarse artifact, cache/data identity and score/ranking consistency. |
| `ablation.py` | Existing suite expansion plus train-stage validation selection and generation of a confirmation suite. |
| `train.py` / `inference.py` / `retrieval.py` | Same model configuration, saved radii, frozen val/test coarse scores and fine-only diagnostics. |
| `evaluation/evaluate.py` | Apply the existing complete-ranking protocol to final, coarse and fine-only rankings. |

Source module paths above are relative to `src/rcr/methods/proposed/` except
`evaluation/evaluate.py`, which is relative to `src/rcr/`.

## Exact interventions

Let `i` be the unit identity embedding from the existing identity head,
`P(i) = W i + b` be the existing query or target identity projection, and
`unit(x) = x / max(||x||_2, 1e-6)`.

| Variant | Identity contribution | Runs |
| --- | --- | --- |
| A — control | `P(i)` | 1 |
| B — fixed pre-projection scale | `P(c * i)`, `c = 4, 8, 16` | 3 |
| C — LayerNorm | `R * unit(LayerNorm(P(i), affine=False))` | 1 |
| D — direction-preserving L2 | `alpha * R * unit(P(i))`, `alpha = 0.5, 1, 2` | 3 |

The same variant is applied on both query and target sides. B retains the
projection bias: `P(c*i)` is not `c*P(i)`. C explicitly renormalizes after
LayerNorm, so its nonzero output norm is R despite LayerNorm epsilon. A zero
vector stays zero; C also maps a constant projected vector to zero. C removes
the coordinate mean; D preserves the direction of the projected identity.
The original identity head, matching, grounding, membership/binding equations,
losses and optimizer settings are unchanged. These modules add no learned
parameters and consume no random initialization draws.

## Fixed initial radii

Calibration runs before the first optimizer update. All variants temporarily
use the initial A forward to measure the same reference. It uses eval mode,
FP32, no gradients, and restores the RNG and training modes afterward.

- `R_q`: mean L2 norm of the initial query role embedding table.
- `R_t`: mean L2 norm of `evidence_proj(evidence) + box_proj(geometry)` over
  **valid target person tokens** in the calibration pairs.
- Default calibration data: the first 128 train samples sorted by `sample_id`,
  each paired with its annotated train `target_image_id`. The IDs are recorded.
  No validation/test samples calibrate R. Padding is excluded, and the target
  average weights tokens rather than giving each differently sized batch equal weight.
- Both scalar radii are registered buffers in checkpoints. They are never
  optimizer parameters, moving averages, or recomputed during retrieval.
- Within a seed, A–D have the same initial trainable parameter hash and R values.
  A different seed gets its own initial R values shared by its control/winner pair.

`initialization.json` records the hash, initial norms, radii and calibration IDs.
The same information is saved in `best.pt` and `last.pt`.

## Fixed shortlist and fusion

The main configuration explicitly fixes:

$$S_{coarse} = z(S_{id}) + 0.4 z(S_{state})$$

$$S_{final} = z(S_{fine}) + 0.4 z(S_{coarse}).$$

Coarse branch z-scores use the full split gallery with the query excluded;
final fusion z-scores use the same Top-500 candidates. Existing finite-score
masking and stable tie ordering are preserved.

**The suite requires one existing reference checkpoint**. By default this is
`runs/proposed/best.pt`, alongside its `tokenizer/` folder. Prefer the checkpoint
used to select your 0.4 coefficients. Set `fixed_coarse.checkpoint` in the suite
if it is elsewhere. This reference is used only to compute shared val/test
coarse rankings and scores, not to initialize the eight training runs.

Before training A, the runner saves the reference coarse outputs for val/test
under `fixed_coarse/`. Every variant and every periodic validation uses the
same actual Top-500 images **and the same coarse scores for final fusion**.
The full coarse tail is retained, so final Full mAP remains the benchmark's
complete-gallery metric. A checksum and cache/split fingerprints prevent stale
or modified artifacts from silently being reused.

Only the variant's fine scores change during these comparisons. This is a
controlled pipeline with a shared reference coarse stage. It does not measure
the effect of letting each retrained model choose a different shortlist.
Train-only diagnostic retrieval and hard-negative mining retain their existing
behavior, schedule and sampling rules; their scores are not used for selection.
Hard-negative pools can evolve differently as the models learn.

## Run

Apply the ZIP files at the repository root. No files need deletion, no dependency
changes are needed, and the frozen visual cache does not need rebuilding.
Use the proposed Python environment; on Kaggle this can be
`.venv-proposed/bin/python` instead of `python`.

1. Update the usual data/cache paths in `configs/methods/proposed.yaml`, and
   point the suite's `fixed_coarse.checkpoint` to the existing reference.
2. Run the eight retraining experiments on complete validation:

```bash
python tools/methods/run.py ablate \
  --config configs/ablations/identity_balance.yaml --splits val
```

Each variant starts from the same seed 0 and initialization. Its `best.pt` is
selected by periodic **pipeline validation Full mAP**, using the frozen
shortlist. The suite then selects among all eight `best.pt` checkpoints by the
same metric. Ties keep YAML order, with A listed first. No test metrics are used
in selection. Reference test scores may be cached beforehand without evaluating
the eight variants on test.

3. Run the generated winner/control confirmation suite, using seeds 1 and 2:

```bash
python tools/methods/run.py ablate \
  --config runs/ablations/identity_balance/confirmation.yaml --splits val test
```

This second command trains four models if another variant beats A, or two if A
wins. It inherits the same frozen coarse artifacts and does not select a new
variant. To report all three seeds on test, also evaluate the already selected
seed-0 winner and seed-0 A, with no retraining:

```bash
python tools/methods/run.py run \
  --config runs/ablations/identity_balance/selected.yaml --splits test
python tools/methods/run.py run \
  --config runs/ablations/identity_balance/A_control/config.yaml --splits test
```

If step 2 uses `--splits val test`, it already evaluates only the chosen variant
and A on test after the validation selection. Do not rerun the whole eight-way
suite to evaluate an existing winner: that command retrains and rebuilds the
reference artifacts. Use `selected.yaml` instead.

For a budget of two confirmation seeds plus seed 0, compare each variant's three
per-seed values and report mean and sample standard deviation. Do not pick the
best test seed. The original eight-run `summary.csv` and confirmation
`summary.csv` retain individual results; they are not pooled across queries.

## Read results

The suite writes `summary.csv`, `selection.json`, `selected.yaml`, and the
ordinary train suite `confirmation.yaml`. Each variant owns `config.yaml`,
`initialization.json`, checkpoints, history, per-epoch evaluation and val/test
outputs. Periodic evaluation still follows the existing epoch frequency.

| Field / file | Meaning |
| --- | --- |
| `full_map`, `full_r1` | Final pipeline after 0.4 fusion; `full_map` selects the winner. |
| `fine_id_r1` | ID R@1 with **raw fine score only** on the exact same shortlist. |
| `fine_full_map`, `fine_full_r1` | Additional fine-only diagnostics with the same coarse tail. |
| `dual_full_map`, `dual_full_r1`, `dual_fine_id_r1` | DUAL-only metrics in the suite summary. |
| `dual_num_queries` | DUAL sample count; interpret small subsets accordingly. |
| `coarse_*`, `candidate_recall_500` | Reference diagnostics; these must match across variants. |
| `norm_query.*`, `norm_target.*` | Token-weighted mean branch norms in the suite summary. |
| `branch_norms.json` | Mean, population standard deviation and count per branch. |
| `R_q`, `R_t`, `initialization_sha256` | Reproducibility checks within each seed. |

Norms include raw identity, unscaled projection, actual added identity, query
role, target evidence, geometry, their sum, and combined tokens. Query counts
exclude absent Subjects/padded people. Target counts cover valid persons over
scored query-candidate pairs. Norms are collected only in calibration/evaluation,
with no per-step training synchronization. DUAL metrics are also present under
`metrics.json -> by_case -> DUAL`. As in the existing evaluator, a case absent
from a split is omitted, and its summary cells are blank; no zero score is
invented for an empty DUAL subset.

Old checkpoints without `model.identity_balance` still load as A. Editing only
the inference YAML never transforms an old checkpoint into B/C/D: model settings
and R come from its training checkpoint. Those variants require retraining.

## Verification and scope

The patch was developed against commit
`3771dbd88da0a51c0d78e8b68e6805f3bc875c1b`. Tests use small local models/caches and
cover the full eight-variant workflow, validation selection, fixed shortlist,
confirmation seeds, saved radii, fine-only/DUAL metrics, gradients, legacy loading
and unchanged initialization. A was also compared directly against that original
commit: all 170 checked weight, loss, diagnostic and gradient tensors matched
exactly on the same two-Subject fixture.

These are software checks, not PIPA accuracy experiments or T4 benchmarks. The
patch contains no claim that B, C or D improves validation performance. Full
training and seed confirmation run in your environment with your cache and
reference checkpoint.
