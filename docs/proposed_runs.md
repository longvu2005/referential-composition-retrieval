# Run guide

Run commands from the repository root using the environments described in the
[script reference](../scripts/README.md#environments).

## Configure paths

Edit [configs/proposed.yaml](../configs/proposed.yaml) for the current machine
and experiment:

| Setting | Purpose |
| --- | --- |
| `data.final_dir`, `data.image_root` | Benchmark annotations and raw images |
| `data.*_cache` | Detection, scene and person feature directories |
| `person_encoder` | Extraction environment and encoder configuration |
| `cache.supervision_cache` | Separate train annotation alignment |
| `output.dir` | Run artifacts |

Model, loss, sampling, retrieval and runtime settings are defined in the same
file. `--set key=value` overrides existing configuration keys.
See [GPU performance](gpu_performance.md) for prefetch, packed feature transfers,
gallery/text budgets, runtime diagnostics and CUDA profiling commands.
Fine evaluation also uses bounded `retrieval.prefetch_batches=1`; set it to
zero for a serial control. Evaluation logs stage times and writes
`timings_seconds` into ranking artifacts and run metadata, so cache/coarse work
can be distinguished from fine context scoring. Calibration reports its own
elapsed CPU time and retains the complete configured coefficient grid.

## Prepare assets and caches

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml
```

Cache construction reuses compatible outputs and resumes incomplete stages.
See the [cache guide](detection_cache.md) for independent builds. Training needs
query features for train and validation; retrieval needs the requested split.

## Train and select a checkpoint

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml
```

Use a fresh `output.dir` for each experiment. For example:

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set train.seed=1 output.dir=runs/proposed-seed1
```

Training performs scheduled validation, calibrates the configured scoring
coefficients and saves the best selection. The selection criterion is defined
in the [benchmark protocol](benchmark_protocol.md).
The aligned person/pair CLS architecture is `fafa-aligned-pair-cls-v5`.
Start a fresh run to use this schedule. Grounding and context
reasoning share one binding for `train.context_warmup_epochs` epochs, including
any validation reasoning. Before mining or training in the following epoch,
reasoning receives a copy of the warmed parameters and buffers. Grounding keeps
its AdamW state, while reasoning starts with fresh state. Zero warmup epochs
split the bindings before epoch 1. A v4 checkpoint supports retrieval using
its shared binding weights.
Resuming a v4 training run requires its original implementation and optimizer
state. Both architectures use the same frozen feature caches.

With the supplied configuration, epochs 1–4 train the base and CLIP text
LoRA with context disabled. Epochs 5–6 add semantic context, and epoch 7 onward
also enables mining and rejection at weight `0.5`. Text LoRA updates the
query and value projections of the last two CLIP text blocks throughout the
25-epoch run. The adapter group uses `optimizer.text_lora_lr=0.0001` and the
same cosine schedule. The frozen CLIP prefix and visual caches can be reused.
Context loss uses the original training coefficients `a_train=0.04` and
`b_train=0.10`; validation calibration changes retrieval coefficients only.
The supplied RCR experiment retains the name `proposed`, uses
`loss.rejection_loss_mode=hinge` and writes to `runs/proposed`.
Configurations missing this key resolve to `softplus`,
preserving the baseline objective. The mode changes only the rejection term;
candidate eligibility, quotas, mining schedule and calibration follow the same
process in both modes.

### Subject member verification ablation

Run the two arms with the same seed, LoRA settings, warmup and mining schedule.
The verification setting changes the semantic loss pairs. The scorer and
rejection objective use the same settings in both arms.

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set loss.context_verify_members=false output.dir=runs/proposed-mask-control
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set loss.context_verify_members=true output.dir=runs/proposed-mask-verified
```

Joint epochs report `train_fine_full_pairs`, `train_fine_partial_pairs` and
`train_fine_active_fraction` for the verified comparisons in both arms.
The active fraction uses the number of sampled query rows as its denominator.
Case suffixes include `_dual` and `_relational`. W&B tracks the same counters
under `epoch/fine_*`.

Compare `fine_unfiltered_pairs` (same-ID with positive coverage) to
`fine_owner_pairs` (correctly owned query crops in each Subject) and
`fine_matched_pairs` (each image verifies at least one member per Subject).
The gap from those matched pairs to `fine_full_pairs + fine_partial_pairs`
comes from unequal verified sets. A full pair verifies every annotated member
on both images. The separate `fine_multi_head_query_crops` and
`fine_multi_head_target_matches` counters flag labeled crops with multiple
annotated heads inside the box. Compare them to the corresponding labeled-crop
and labeled-match counts.

### Rejection loss ablation

For a comparison of rejection modes, run both from the same source commit with
seed `0` and the same caches/runtime. These commands retain rejection weight `0.5`,
temperature `0.1`, `retrieval.context_weight=0.10`, partial-ID quota `6` and
mining context coefficient `0.005` from the supplied configuration.

Run each command in a separate clean workspace at that commit. Both keep the
experiment and W&B name `proposed` and use `runs/proposed` within their workspace.

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set loss.rejection_loss_mode=softplus train.seed=0
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set loss.rejection_loss_mode=hinge train.seed=0
```

Each workspace must have a fresh `runs/proposed`. Compare validation-selected
metrics and the rejection/semantic loss histories in each run; raw rejection magnitudes
have different meanings across the two objectives. The `0.005` coefficient is
used only to mine candidates; `0.10` is `b_train` in the loss. Validation still
calibrates `b` for inference. Changing the mode requires a new run; the existing
resume checks enforce the effective configuration and source contract.

The supplied experiment disables verified left/right swap supervision
(`loss.context_swap_weight=0.0`). To enable this optional objective, use a
separate run directory, for example:

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set loss.context_swap_weight=0.2 output.dir=runs/proposed-with-swap
```

`train.sampling.partial_id_quota=6` starts at epoch 1 for queries requiring at
least two distinct IDs, including one-Subject GROUP queries. Rows keep 32 unique
candidates: Full Positives, the existing same-ID semantic quota, mined wrong-ID
negatives, a partial-ID top-up, then random fallback. The target of six includes
partial-ID images already drawn by mining. The top-up prioritizes four images
missing one ID and reserves absent-Subject and missing-member representatives
within the remaining budget. With two singleton Subjects, it covers both
omissions; single-ID choices favor IDs missing less often in the current row.
Remaining diversity slots draw overlap groups with probability proportional
to `2^(r-1)`. Images
containing every requested ID plus extra people remain identity positives.

Mining refreshes at epochs 7, 11, 15, 19 and 23 in the supplied schedule. Base
semantic and wrong-ID pools use H=0 with population normalization over their
screened union. With context and rejection active,
`train.sampling.rejection_mining_context_weight=0.005` also scores H on at most
`pool_size=64` screened wrong-ID images per query.
Population fusion on that same wrong-ID list is computed before filtering
`C>0`. Up to two images that improve rank and enter the H top ten form the
rejection context pool. Each row randomly draws one of them before two base
wrong-ID slots; unavailable slots return to the base pool, then random fallback.
Only actual mined wrong-ID selections enter the rejection mask. Extra partial-ID
and random selections retain their GT labels.

Old configurations resolve the two new sampling keys to `0` and `0.0`, preserving
base sampling and its RNG sequence. Changing either setting requires a new
`output.dir`; resume compatibility checks retain the existing source/config
contract. Mid-epoch resume restores hard pools, including the rejection context
pool, and committed RNG states without refreshing them. `hard_negatives.json`
records the mining coefficient separately from training/inference weights.
`history.jsonl` logs `train_candidate_partial_id` (all selected partial-ID images,
including mining and random draws) and `train_candidate_rejection_context`
(actual H-slot selections).

W&B tracking is configured by `wandb`; use `wandb.mode=offline` or
`wandb.enabled=false` when online tracking is unavailable.

## Resume training

Continue the same run using its saved effective configuration:

```bash
.venv-proposed/bin/python tools/run.py train --config runs/proposed/config.yaml --resume
```

An explicit path may follow `--resume`. Resume uses the full training state in
`last.pt`, including optimizer, random generators and sampling progress. Keep
the original run directory, numerical settings, total epoch count, source,
runtime, data and caches compatible with the saved state. `best.pt` is the
inference checkpoint.
The saved `binding_shared` flag restores the module sharing and optimizer
parameter order. Resume within joint training preserves the learned reasoning
weights. A resumed warmup run splits at the configured boundary.
`model.fine_matching.owner_mode` is part of the effective configuration for
resume.

## Seeds and reproducibility

`train.seed` controls random initialization and sampling.
`runtime.deterministic` requests reproducible kernels. Record the source
revision, effective configuration, runtime and data/cache identities with each
experiment; hardware and software changes can affect numerical results.

## Inspect validation and optional calibration

Training writes the selected validation result automatically. Export it with:

```bash
.venv-proposed/bin/python tools/report.py --split val
```

Optional offline calibration searches saved validation scores:

```bash
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
```

Configure validation artifacts in
[configs/calibration.yaml](../configs/calibration.yaml). Calibration reuses saved
scores on CPU and exports a selected configuration for test.

## Run test

```bash
.venv-proposed/bin/python tools/run.py run \
  --config runs/proposed/selected.yaml --split test
```

Test uses the validation-selected checkpoint and scoring settings. The runner
checks checkpoint, dataset and cache compatibility before retrieval.

## Compare Subject owner rules

`model.fine_matching.owner_mode` defaults to `raw_logit`, including for older
configs and inference checkpoints without this field. Run both modes on the
same validation checkpoint with its selected scoring coefficients.

```bash
for owner_mode in raw_logit crop_log_softmax; do
  diagnostic_dir="runs/owner-diagnostic-${owner_mode}"
  .venv-proposed/bin/python tools/run.py retrieve \
    --config runs/proposed/selected.yaml --split val \
    --diagnostic-owner-mode "$owner_mode" --set "output.dir=${diagnostic_dir}"
  .venv-proposed/bin/python tools/run.py evaluate \
    --config runs/proposed/selected.yaml --split val \
    --set "output.dir=${diagnostic_dir}"
done
```

The diagnostic recomputes fine evidence using the same checkpoint and coarse
shortlist. Compare by-case Full-mAP in `metrics.json`, especially DUAL and
RELATIONAL. A diagnostic using a mode different from the checkpoint is marked
as an ablation. Calibration uses artifacts from the checkpoint's trained mode.

To compare the effect on learning, train each mode with the same seed and
caches in its own run directory. Both runs use the method name `proposed`.

```bash
for owner_mode in raw_logit crop_log_softmax; do
  .venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
    --set "model.fine_matching.owner_mode=${owner_mode}" train.seed=0 \
    "output.dir=runs/proposed-owner-${owner_mode}"
done
```

`best.pt`, `last.pt`, `selected.yaml` and retrieval artifacts record the mode.
Use the selected configuration from the corresponding run for test.

## Reading outputs

Paths below are relative to the configured `output.dir`.

| Artifact | Contents |
| --- | --- |
| `config.yaml`, `runtime.json` | Effective settings and runtime metadata |
| `history.jsonl` | Training and validation history |
| `last.pt` | Full training state for resume |
| `best.pt`, `selected.yaml`, `selection.json` | Selected model and scoring settings |
| `evaluation/` | Scheduled validation summaries |
| `selected/val/`, `selected/test/` | Rankings, run metadata and evaluated metrics |

See the [reporting guide](experiments.md) for evaluation and table export.

## Score artifacts and CPU replay

`rankings.pt` contains rankings, shortlist evidence and provenance. Compatible
artifacts can be replayed without model weights or feature caches:

```bash
.venv-proposed/bin/python -m rcr.proposed.replay \
  runs/proposed/selected/val/rankings.pt --output-dir runs/replayed/val
```

Replay can vary supported score coefficients on the saved shortlist. Changes
to the model, matching, query inputs or shortlist require fresh retrieval.
Use validation to select coefficients under the benchmark protocol.
Retrieval stores raw `H` before its coefficient. Replay uses the fine evidence
already saved for that owner mode. Compatible artifacts from before the owner
setting use `raw_logit`.
