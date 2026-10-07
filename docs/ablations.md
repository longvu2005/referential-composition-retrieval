# Ablation for Person Identity–Semantic Modeling

The previous ablation set has been removed. This version focuses on the backbone, representation/routing, and binding score. Use the same data/splits, sampling strategy, training schedule, and seed when making comparisons.

## 1. Calibration of the main model

This step selects inference coefficients; it is **not an architectural ablation**:

```bash
.venv-proposed/bin/python tools/methods/run.py ablate \
  --config configs/calibration/joint.yaml --splits val test
```

An 8 × 7 grid jointly selects `coarse_beta` and `fine_coarse_weight` using **final Full-mAP on the validation set**, at Top-500. Only the winning coefficient pair is evaluated on the test set. The selected coefficients and the checkpoint/validation fingerprint are stored in `runs/calibration/selected.yaml`.

Do not select models or coefficients using the test set. Re-run calibration whenever a new checkpoint is used.

## 2. Binding: one checkpoint, same shortlist

Run this after calibration:

```bash
.venv-proposed/bin/python tools/methods/run.py ablate \
  --config configs/ablations/binding.yaml --splits val test
```

| Variant | Explicit person binding | Components that remain active |
| --- | --- | --- |
| `full` | max_j(ID + semantic), followed by soft membership pooling | Semantic context and coarse ID/global state |
| `without_identity_binding` | max_j(semantic) | Coarse retrieval still uses identity |
| `without_semantic_binding` | max_j(ID) | Context still receives person semantics/instruction |
| `context_only` | 0 | Context reasoner and coarse retrieval |

This is an **inference-time term removal**, not removal of the entire head. All variants use the same checkpoint, gamma, coarse scores/order/Top-M, and the selected fusion coefficients. Query features, target context, and pair scores are reused; the fine network is not run separately for each variant. Do not retune coefficients for individual term-removal variants in this table.

Results are written to `runs/ablations/binding/summary.csv`, together with rankings and metrics for each variant/split. Inspect ID-mAP, Full-mAP, CandidateRecall@500, and per-case results. Coarse metrics must remain identical; fine/full metrics reflect the contribution of each score term.

## 3. Backbone × representation: train from scratch

The same DINO scene/detection cache is shared by every variant. It also holds the
DINO crop features for the DINO controls. FAFA variants add the separate person
cache built by the main pipeline. Reuse your existing DINO cache or build it once:

```bash
.venv-proposed/bin/python tools/methods/run.py build-cache \
  --config configs/methods/proposed.yaml --cache-stage dino

.venv-proposed/bin/python tools/methods/run.py ablate \
  --config configs/ablations/representation.yaml --splits val test
```

| Variant | Person backbone | Representation/routing |
| --- | --- | --- |
| `dino_shared` | DINO | A shared head feeds identity, composition, and target-token addition |
| `fafa_shared` | FAFA image-only pooled hidden tokens | Same shared-head control |
| `dino_dual` | DINO | Two separate heads + identity detach + same-person binding |
| `fafa_dual` | FAFA image-only pooled hidden tokens | Main architecture |

`shared` is a control that is retrained using the previous shared structure; it does not reuse an old checkpoint.

All variants use the same seed, schedule, sampling strategy, loss weights, and default inference coefficients of 0.4/0.4; no variant-specific calibration is performed for this comparison. Each run selects its epoch using the same validation rule and has its own checkpoint.

Backbone pairs keep the architecture identical. The shared-vs-dual comparison measures the entire representation/routing/binding package; it does not isolate the effect of the number of projection heads alone.

Each variant generates its own shortlist using its own model. Therefore, inspect CandidateRecall@500 to distinguish changes in coarse retrieval from changes in fine ranking. Do not interpret this table as a comparison of fine ranking under a fixed shortlist. The binding experiments in Section 2 explicitly control the shortlist.

DINO variants automatically use `data.dino_cache` from the method YAML as their
`data.cache`. An explicit experiment override of `data.cache` takes precedence.
FAFA variants inherit both paths from the main config. Keep
`cache.allow_legacy_dino=true` for the old Kaggle source; no detector is rerun.

You may add:

```yaml
sweep: {train.seed: [0, 1, 2]}
```

to each experiment to run paired seeds. Run names automatically include the sweep value. Matching initialization hashes are not required between architectures with different parameter counts or parameter shapes.

## 4. Optional: retrain the binding variants

```bash
.venv-proposed/bin/python tools/methods/run.py ablate \
  --config configs/ablations/binding_train.yaml --splits val test
```

This suite trains four FAFA/dual models (`both`, `identity`, `semantic`, `none`) using the same cache, schedule, seed, and default inference coefficients.

It tests whether the model can adapt when one binding term is absent, which is different from removing a term from a checkpoint that was trained with all terms enabled, as in Section 2.

Semantic context and coarse identity remain active in all four variants.

## Limitations and interpretation of results

- The FAFA cache mean-pools hidden tokens; it does not reproduce the native multi-token FDA representation.
- Binding retains soft membership and an optimistic max operation; it does not yet use one-to-one assignment or guarantee that all GROUP members are covered.
- Separate dual heads and contrastive supervision do not by themselves guarantee clothing invariance.
- Local tests verify gradients, masks, cache behavior, and execution flow; PIPA quality, CUDA T4 behavior, and FAFA cache-building cost must be measured in the actual experimental environment.
- The current test export contains only 14 queries. Report results by case and avoid treating fluctuations of only a few queries as general conclusions.
