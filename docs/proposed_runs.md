# Proposed method: experiment protocol

Run every command from the repository root. Use Python 3.11 or 3.12 and keep the
proposed (PyTorch >=2.6) and FAFA (PyTorch 2.5.1) environments separate.

## 1. Configure paths

Edit `configs/proposed.yaml`:

```yaml
data:
  final_dir: dataset/data/final
  image_root: /path/to/images
  scene_cache: /path/to/clip-b32-scene
  fafa_image_cache: /path/to/fafa-images
  fafa_query_cache: /path/to/fafa-queries
person_encoder:
  python: .venv-fafa/bin/python
```

All model, optimizer, sampling, retrieval and runtime choices live in this one
file. Command-line overrides must name an existing key.

## 2. Prepare frozen assets and caches

```bash
python tools/run.py prepare --config configs/proposed.yaml
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage scene
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-image
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query
```

`build-cache` without `--cache-stage` runs all missing stages. CLIP owns scene/global
features and crop geometry. The FAFA image cache owns query-independent person
means and target tokens. The FAFA query cache owns text-conditioned query
features. Editing query text therefore invalidates only the last stage.

The FAFA worker runs through `person_encoder.python`, so native FAFA never enters
the proposed environment. Training and retrieval read cache files and do not load
FAFA weights.

## 3. Train and select a checkpoint

```bash
python tools/run.py train --config configs/proposed.yaml
```

The training loss remains raw `alpha * B_FAFA + coverage * context`. Periodic
validation ranks with Context z-score, fixed `context_weight=0.0325`; it normalizes
only after scoring a query's complete Coarse Top-M. DINO/BERT checkpoints are
incompatible; train a fresh CLIP checkpoint. To reproduce the previous fine
ranking, use `--set retrieval.fine_score_mode=legacy_raw` at retrieval time.

Training uses only train annotations. Before epoch 1, GT heads are aligned to the
fixed detector crops required by train queries and their Full Positives. The
alignment is cached separately from visual features.

Each batch contains:

- up to `positives_per_query` Full Positives;
- `identity_fraction` same-identity, wrong-state negatives;
- `hard_fraction` coarse hard negatives after the configured warm-up;
- uniform negatives for the remaining slots.

The identity loss can additionally read up to
`train.identity_sampling.max_crops` detector-aligned train crops. These crops do
not enter the candidate set, FAFA scoring or state loss; they only keep the
supervised contrastive objective supplied with cross-image positives.

The model is evaluated on the complete validation split at
`evaluation.every_epochs` and on the final epoch. `best.pt` maximizes

```text
0.5 * overall Full-mAP + 0.5 * macro-case Full-mAP.
```

Use a separate output directory for every seed or ablation:

```bash
python tools/run.py run --config configs/proposed.yaml --train --splits val \
  --set train.seed=1 output.dir=runs/proposed-seed1
```

W&B tracking is configured by the `wandb` block in `configs/proposed.yaml`.
Each run records only research configuration and cache provenance, not local
dataset paths. It logs losses, gradient norm and learning rates every
`log_every_steps`, then epoch means and validation metrics. Set `mode: offline`
to sync later with `wandb sync`, or set `enabled: false` for untracked smoke
tests.

## 4. Calibrate retrieval weights

```bash
python tools/run.py calibrate --config configs/calibration.yaml
```

Calibration reads the saved training config at
`runs/proposed-clip-b32/config.yaml`, including actual paths and training overrides.
It shares model scores across the configured `(coarse_beta,
fine_coarse_weight)` grid and evaluates every pair on validation. It writes the
selected frozen configuration under `runs/clip-b32-calibration/`. Matching thresholds and
the missing penalty are architecture hyperparameters; this command does not tune
them.

`retrieval.fine_score_mode` and `retrieval.context_weight` come from `model_config`
and remain fixed throughout this grid (defaults: `context_zscore`, `0.0325`).
`selected.yaml` includes these values, the selected pair and Top-M in
`selected_retrieval`, plus checkpoint SHA256 and validation fingerprint. The test
path rejects changed scoring settings; inference chunk sizes can still change.
Selected YAMLs produced before score schema 1 must be regenerated on validation
so their weights cannot silently carry over to a different Fine scoring mode.

## 5. Run the frozen test configuration

```bash
python tools/run.py run \
  --config runs/clip-b32-calibration/selected.yaml \
  --splits test
```

The runner verifies the checkpoint hash, validation fingerprint, feature cache
identity and encoder metadata before ranking. Test labels are used only by the
evaluator.

To publish the selected validation result before test:

```bash
python tools/run.py run --config runs/clip-b32-calibration/selected.yaml --splits val
```

## Kaggle reuse

To resume a FAFA image-cache export from a Kaggle notebook, use the downloader
from the FAFA environment with explicit notebook coordinates:

```bash
.venv-fafa/bin/python download_fafa.py \
  --output cache/clip-b32-fafa-images --user OWNER --kernel NOTEBOOK --version VERSION
```

Authentication uses the normal Kaggle client configuration. The downloader
records its source in the resume state and refuses to continue that state with
different notebook arguments.

CLIP uses new `data.scene_cache`, `data.fafa_image_cache` and
`data.fafa_query_cache` paths. Rebuild all three stages and train a fresh model;
old DINO/BERT checkpoint and cache identities must not be substituted. Keep
raw images available for the frozen FAFA crop extraction. Once new caches are
complete, mounted read-only copies are supported by training/retrieval.
See [CLIP migration](clip_b32_migration.md) for the exact commands and text policy.

## Reading outputs

`history.jsonl` contains one compact row per epoch. Each validation directory
contains overall and by-case metrics. Split retrieval output additionally stores
coarse, fine, fused and FAFA diagnostic rankings. The FAFA diagnostic reranks the
same coarse Top-M with FDA binding only; it is not the standalone FAFA baseline.
Checkpoints are inference/reproducibility artifacts, not resumable trainer states:
they omit optimizer/scaler state and store only the learned text projection,
native span policy and frozen CLIP revision.

For a fair ablation, change one owned component, keep data/evaluation fixed, and
record the commit, full YAML, seed, runtime and validation result.

## Score artifacts and CPU replay

Every score vector is aligned with `coarse_topm`, **before** Fine sorting. Let N
be the query count, G the gallery size and M = min(requested Top-M, G-1).

| Key | Type/shape | Meaning |
| --- | --- | --- |
| `sample_ids`, `gallery_ids` | lists of strings | Ordered public identifiers |
| `rankings` | int32 `[N,G-1]` | Fused Top-M followed by unchanged Coarse tail |
| `coarse_rankings` | int32 `[N,G-1]` | Actual complete Coarse ordering |
| `coarse_topm` | int32 `[N,M]` | Gallery indices defining all score columns |
| `fine_rankings` | int32 `[N,G-1]` | Selected Fine mode plus Coarse tail |
| `fafa_rankings` | int32 `[N,G-1]` | Raw native FAFA binding plus Coarse tail |
| `fafa_binding` | float32 `[N,M]` | Raw native binding B |
| `context` | float32 `[N,M]` | Raw reasoner output |
| `coverage` | float32 `[N,M]` | Matching coverage C |
| `coarse_topm_scores` | float32 `[N,M]` | Actual Coarse scores for the selected beta |
| `gated_context` | float32 `[N,M]` | C times raw context |
| `fine_raw_scores` | float32 `[N,M]` | alpha B + gated context |
| `fine_scores` | float32 `[N,M]` | Selected Fine score before final fusion |
| `fused_topm_scores` | float32 `[N,M]` | Final shortlist score |
| `context_mean`, `context_std` | float32 `[N]` | Gated context population statistics |
| `score_schema_version` | int | 1 |
| `binding_scale` | float | Learned checkpoint alpha |
| `score_config` | primitive dictionary | Mode, context/coarse/fusion weights, requested Top-M, alpha, epsilon, population/FP32/CPU normalization, alignment and tie/invalid policies |

By default, only computed Fine candidates are serialized. Fine evidence must be finite;
unsupported Coarse values retain their non-finite representation and are masked
by positive-weight final fusion. `run.json` records the same score configuration,
checkpoint epoch/hash, dataset/validation fingerprints, commit and dirty state.
The evaluator ignores diagnostic score fields and still reads old artifacts.
The optional internal `ranking_limit` produces incomplete rankings for diagnostics
only; replay and official evaluation require the complete Coarse order.

Install the repository and CPU evaluation dependencies once. Replay then needs
only `rankings.pt`, with no checkpoint, GPU, native libraries or feature cache:

```bash
# Reproduce the recorded Proposed fused ranking exactly.
python -m rcr.proposed.replay runs/clip-b32-calibration/selected/val/rankings.pt \
  --output-dir runs/replay-exact/val

# Legacy Fine only: alpha B + C * context.
python -m rcr.proposed.replay runs/clip-b32-calibration/selected/val/rankings.pt \
  --output-dir runs/replay-legacy/val --mode legacy_raw --fine-coarse-weight 0

# Raw Gamma: alpha B + gamma * C * context.
python -m rcr.proposed.replay runs/clip-b32-calibration/selected/val/rankings.pt \
  --output-dir runs/replay-gamma/val --mode raw_gamma \
  --context-weight 0.0325 --fine-coarse-weight 0

# Context z-score, with optional Coarse fusion on the same saved shortlist.
python -m rcr.proposed.replay runs/clip-b32-calibration/selected/val/rankings.pt \
  --output-dir runs/replay-context/val --mode context_zscore \
  --context-weight 0.0325 --fine-coarse-weight 0.4

# Both z-score ablation: z(alpha B) + lambda * z(C * context).
python -m rcr.proposed.replay runs/clip-b32-calibration/selected/val/rankings.pt \
  --output-dir runs/replay-both/val --mode both_zscore \
  --context-weight 0.0325 --fine-coarse-weight 0
```

Omitted arguments reuse recorded values. `--fine-coarse-weight 0` yields Fine
only. Positive values apply the same final fusion helper as inference. Replays
keep the Coarse tail and carry source hash/provenance into a separate output
directory. For evaluation, use `tools/run.py evaluate` with the same split and
data configuration and `--set output.dir=runs/replay-context` (the evaluator
appends `/val` or `/test`).

Old/missing/incompatible score schemas fail explicitly for replay; ranks cannot
recover actual Coarse scores. These Top-M-only artifacts cannot replay a `coarse_beta`
that changes shortlist membership: newly admitted candidates lack Fine evidence.
Do any offline parameter selection on validation; freeze it before test.

## Full-gallery tuning export

Use the existing trained checkpoint and caches to make one validation export:

```bash
python tools/run.py run --config runs/proposed-clip-b32/config.yaml \
  --splits val --save-tuning --set output.dir=runs/proposed-clip-b32-tuning
```

`retrieve --save-tuning` is also available when evaluation will be run separately.
The flag is Proposed/validation only. It never triggers training and is not
passed to periodic training validation. If combined with `run --train`, only the
standalone retrieval after checkpoint selection performs the full export.
Use a new output root to keep the original run together.

Ordinary Coarse Top-M forwards are performed first. The exporter then computes
the remaining non-self Fine pairs, once per query/target, in the same bounded
`fine_batch_size` chunks. This is necessary: a different beta can admit any
previously unscored gallery image. It increases Fine pair count from approximately
`N*M` to `N*(G-1)` for `N` queries and `G` gallery images. No raw images or native
backbone extraction are repeated when the caches are complete.

`rankings.pt` keeps score schema 1 and adds an independent `tuning_scores`
dictionary; the evaluator's existing ranking/positive/metric protocol is unchanged:

| Field inside `tuning_scores` | Format | Meaning |
|---|---|---|
| `schema_version`, `scope`, `score_alignment` | `1`, `full_gallery`, `gallery_ids` | Versioned complete-gallery contract |
| `query_indices` | int64 `[N]` | Self index for each query in this exact ordered gallery |
| `identity_scores` | float32 `[N,G]` | Raw complete-gallery Coarse identity branch |
| `state_scores` | float32 `[N,G]` or `None` | Raw state branch, saved even if the original beta was zero, when trained |
| `fafa_binding`, `context`, `coverage` | float32 `[N,G]` each | Raw Fine components before gating/normalization, for every non-self pair |
| `state_trained` | bool | An untrained state branch remains disabled during replay |
| `fixed_fine_matching` | primitive dictionary | Matching thresholds, missing penalty and duplicate policy used in this forward |
| `normalization_dtype`, `normalization_device`, `eps`, `std_unbiased` | FP32, CPU, `1e-6`, `false` | Same arithmetic contract as direct retrieval |
| `provenance` | primitive dictionary | Checkpoint/cache identifiers and benchmark/split fingerprints, embedded by standalone retrieval |

Fine self entries are zero sentinels and are never ranked. Identity/state retain
their original non-finite values; eligibility and self exclusion are recomputed
for each beta by the shared `combine_scores` helper. Coarse normalization/sorting
and shortlist Fine normalization/sorting run on CPU FP32 in both direct retrieval
and replay. Model forwards still use the configured device. As before, changing
model-forward batch shapes may produce small floating point differences; replay
uses the actual exported values and needs no forwards.

The extra raw arrays occupy approximately `20*N*G` bytes when state is trained,
about 28.6 MiB at 264 queries / 5,684 images, plus IDs/metadata. They are stored
once per file, rather than duplicating features/tokens/checkpoints. Keep all
`rankings.pt` bytes when copying the export; logs/configs alone do not contain
the score matrices.

Now change beta, requested Top-M, context weight and final fusion weight offline:

```bash
python -m rcr.proposed.replay runs/proposed-clip-b32-tuning/val/rankings.pt \
  --output-dir runs/replayed/val --coarse-beta 0.2 --top-m 500 \
  --context-weight 0.0288 --fine-coarse-weight 0.0064
python tools/run.py evaluate --config runs/proposed-clip-b32/config.yaml \
  --splits val --set output.dir=runs/replayed
```

These numbers illustrate the CLI; the export does not select an optimal beta.
Replay rebuilds the whole Coarse ranking and its tail, gathers raw Fine evidence
for the new shortlist, recomputes that shortlist's statistics, and regenerates
coarse/Fine/FAFA/fused diagnostic rankings. The full-gallery matrices survive
successive replays, including after reducing Top-M. Stable Coarse ties keep
gallery order; shortlist ties keep the new Coarse order. An old/incomplete
artifact fails explicitly if asked to change beta or Top-M.

Keep evaluation `candidate_ks` no larger than the requested Top-M; reduce them
in the evaluation config when testing a smaller shortlist. Validation labels
belong only to the evaluator/weight selection. A single replay does not export a
frozen test-selection config and removes stale selection guards from copied
config metadata. Use offline calibration on the same export to freeze weights:

```bash
python tools/run.py calibrate --config configs/calibration_saved.yaml
```

`configs/calibration_saved.yaml` points to the original run config, exported
`rankings.pt` and an output root. It accepts `coarse_beta`, `fine_coarse_weight`,
optional `context_weight` and optional `top_m` grids. The mode stays fixed in
`model_config`. No checkpoint, native library, backbone weights or visual cache
is loaded; only the score file and benchmark annotations are read. Exported
checkpoint hash and validation fingerprint are required, and partial query
exports or changed text/labels/ordered gallery are rejected before publication.
The objective is the shared evaluator's overall Full-mAP, with the first YAML
grid tuple winning ties. This is validation selection, not an independent test
estimate.

Offline calibration writes `summary.csv`, `selection.json`, `selected.yaml` and
the best `selected/val/{rankings.pt,run.json,metrics.json}`. It does not serialize
one full-gallery artifact per grid tuple. Candidate-recall cutoffs are limited
to each tested shortlist; the selected config retains suitable cutoffs.
`selected.yaml` includes the same checkpoint/validation/scoring guards as online
calibration and can be passed to the standard test retrieval command. The real
checkpoint and caches are needed only when subsequently running that new test
split.

This supports score-weight/normalization/Top-M studies with fixed model weights,
gallery, detections, matching rules and query text. Changing identity/member
thresholds, missing penalties, FDA-k, the model or inputs still requires new
forwards. Previously created Top-M-only artifacts cannot be upgraded without
computing their missing Fine evidence once; no retraining is needed.
