# Proposed method: experiment protocol

Run every command from the repository root. Use Python 3.11 or 3.12 and keep the
proposed and FAFA environments separate.

## 1. Configure paths

Edit `configs/proposed.yaml`:

```yaml
data:
  final_dir: dataset/data/final
  image_root: /path/to/images
  dino_cache: /path/to/dino-cache
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
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage dino
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-image
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query
```

`build-cache` without `--cache-stage` runs all missing stages. DINO owns scene
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
validation ranks with Context z-score, fixed `context_weight=0.03`; it normalizes
only after scoring a query's complete Coarse Top-M. Existing checkpoints retain
their architecture and learned binding scale. To reproduce the previous fine
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

Calibration shares model scores across the configured `(coarse_beta,
fine_coarse_weight)` grid and evaluates every pair on validation. It writes the
selected frozen configuration under `runs/calibration/`. Matching thresholds and
the missing penalty are architecture hyperparameters; this command does not tune
them.

`retrieval.fine_score_mode` and `retrieval.context_weight` come from `model_config`
and remain fixed throughout this grid (defaults: `context_zscore`, `0.03`).
`selected.yaml` includes these values, the selected pair and Top-M in
`selected_retrieval`, plus checkpoint SHA256 and validation fingerprint. The test
path rejects changed scoring settings; inference chunk sizes can still change.
Selected YAMLs produced before score schema 1 must be regenerated on validation
so their weights cannot silently carry over to a different Fine scoring mode.

## 5. Run the frozen test configuration

```bash
python tools/run.py run \
  --config runs/calibration/selected.yaml \
  --splits test
```

The runner verifies the checkpoint hash, validation fingerprint, feature cache
identity and encoder metadata before ranking. Test labels are used only by the
evaluator.

To publish the selected validation result before test:

```bash
python tools/run.py run --config runs/calibration/selected.yaml --splits val
```

## Kaggle reuse

To resume a FAFA image-cache export from a Kaggle notebook, use the downloader
from the FAFA environment with explicit notebook coordinates:

```bash
.venv-fafa/bin/python download_fafa.py \
  --output cache/fafa-images --user OWNER --kernel NOTEBOOK --version VERSION
```

Authentication uses the normal Kaggle client configuration. The downloader
records its source in the resume state and refuses to continue that state with
different notebook arguments.

For the published legacy DINO cache, point `data.dino_cache` to the mounted
directory containing `index.pt` and `features/`, keep
`cache.allow_legacy_dino: true`, and put both FAFA caches in writable storage.
Raw images remain necessary because FAFA extracts the detector crops.

When all three mounted caches are already complete and their paths configured:

```bash
cd /kaggle/working/referential-composition-retrieval
python -m pip install -e . --no-deps
python tools/run.py prepare-supervision --config configs/proposed.yaml
python tools/run.py train --config configs/proposed.yaml \
  --set output.dir=runs/proposed-context-z003 \
        checkpoint=runs/proposed-context-z003/best.pt
python tools/run.py calibrate --config configs/calibration.yaml \
  --set model_config=runs/proposed-context-z003/config.yaml \
        output_dir=runs/calibration-context-z003
```

Calibration reads the saved training config, including the explicit checkpoint
path above. Its selected config lives in `runs/calibration-context-z003/selected.yaml`;
use that path for subsequent `run --splits val` and `run --splits test` commands.
Use W&B online/offline settings as appropriate for the Kaggle session.

The legacy cache has no encoder metadata or pixel boxes. The code checks its
gallery order and patch grid, then reconstructs pixel boxes by inverting the
original letterbox transform. Keep the original DINOv3 ViT-B/16, 224 x 224 scene,
256 x 128 person, and Grounding DINO tiny settings.

## Reading outputs

`history.jsonl` contains one compact row per epoch. Each validation directory
contains overall and by-case metrics. Split retrieval output additionally stores
coarse, fine, fused and FAFA diagnostic rankings. The FAFA diagnostic reranks the
same coarse Top-M with FDA binding only; it is not the standalone FAFA baseline.
Checkpoints are inference/reproducibility artifacts, not resumable trainer states:
they omit optimizer/scaler state and store only the learned text projection,
Subject-marker embeddings and frozen BERT revision.

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

Only computed Fine candidates are serialized. Fine evidence must be finite;
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
python -m rcr.proposed.replay runs/calibration/selected/val/rankings.pt \
  --output-dir runs/replay-exact/val

# Legacy Fine only: alpha B + C * context.
python -m rcr.proposed.replay runs/calibration/selected/val/rankings.pt \
  --output-dir runs/replay-legacy/val --mode legacy_raw --fine-coarse-weight 0

# Raw Gamma: alpha B + gamma * C * context.
python -m rcr.proposed.replay runs/calibration/selected/val/rankings.pt \
  --output-dir runs/replay-gamma/val --mode raw_gamma \
  --context-weight 0.03 --fine-coarse-weight 0

# Context z-score, with optional Coarse fusion on the same saved shortlist.
python -m rcr.proposed.replay runs/calibration/selected/val/rankings.pt \
  --output-dir runs/replay-context/val --mode context_zscore \
  --context-weight 0.03 --fine-coarse-weight 0.4

# Both z-score ablation: z(alpha B) + lambda * z(C * context).
python -m rcr.proposed.replay runs/calibration/selected/val/rankings.pt \
  --output-dir runs/replay-both/val --mode both_zscore \
  --context-weight 0.03 --fine-coarse-weight 0
```

Omitted arguments reuse recorded values. `--fine-coarse-weight 0` yields Fine
only. Positive values apply the same final fusion helper as inference. Replays
keep the Coarse tail and carry source hash/provenance into a separate output
directory. For evaluation, use `tools/run.py evaluate` with the same split and
data configuration and `--set output.dir=runs/replay-context` (the evaluator
appends `/val` or `/test`).

Old/missing/incompatible score schemas fail explicitly for replay; ranks cannot
recover actual Coarse scores. These artifacts cannot replay a `coarse_beta`
that changes shortlist membership: newly admitted candidates lack Fine evidence.
Do any offline parameter selection on validation; freeze it before test.
