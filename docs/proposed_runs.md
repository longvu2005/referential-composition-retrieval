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

## 5. Run the frozen test configuration

```bash
python tools/run.py run \
  --config runs/calibration/selected.yaml \
  --splits test
```

The runner verifies the checkpoint hash, validation fingerprint, feature cache
identity and encoder metadata before ranking. Test labels are used only by the
evaluator.

## Kaggle reuse

For the published legacy DINO cache, point `data.dino_cache` to the mounted
directory containing `index.pt` and `features/`, keep
`cache.allow_legacy_dino: true`, and put both FAFA caches in writable storage.
Raw images remain necessary because FAFA extracts the detector crops.

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
