# Referential Composition Retrieval

Research code and benchmark for **Referential Composition Retrieval (RCR)**.
Each query combines a reference image, text identifying one or more Subjects,
and a requested change. Retrieval ranks images that preserve those identities
and satisfy the change.

```text
final dataset -> frozen feature caches -> train -> retrieve -> evaluate
```

## Method

Frozen CLIP ViT-B/32 encodes scenes and text; frozen FAFA encodes person crops.
The learned RCR modules ground Subject descriptions, compose the requested
change, and reason over target evidence. Coarse retrieval uses identity and
global state scores. Fine retrieval uses a partial one-to-one identity assignment
with null matches, then reranks only the coarse Top-M:

```text
S_fine = alpha * B_FAFA + a * coverage + b * context
S_final = Z_M(S_fine) + eta * Z_M(S_coarse)
```

For positive `eta`, `Z_M` uses population statistics on the same finite coarse
Top-M support. Constant branches contribute zero; `eta=0` returns raw fine scores.

`alpha=1` is fixed. Semantic context loss ranks `b * context`; identity rejection
ranks the complete fine score, both with `loss.context_temperature`. Hard mining
uses binding, coverage and coarse fusion with context weight fixed at zero.
Each scheduled validation retrieves once and searches all 729
`(a, b, eta)` triples on CPU. `best.pt` maximizes the mean of overall and macro-case
validation Full-mAP after calibration. Test freezes the winning checkpoint and
coefficients together. Stable ties retain coarse order; the gallery tail is unchanged.

See the [method and equations](docs/proposed_method.md),
[feature contracts](docs/feature_contracts.md) and
[experiment protocol](docs/proposed_runs.md) for objectives, sampling, score
normalization, calibration and compatibility rules. The current experiment
settings are in [configs/proposed.yaml](configs/proposed.yaml).

## Setup

Use Python 3.11 or 3.12. Training requires PyTorch >=2.6 for the pinned CLIP
checkpoint; frozen FAFA extraction uses PyTorch 2.5.1 in a separate environment:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash person-encoder
```

Setup can be rerun to complete an interrupted installation. See the
[environment guide](scripts/README.md#environments) for interpreter selection,
Kaggle setup and optional dataset/evaluation environments.

Raw images are not committed. Follow the [dataset guide](dataset/README.md), then
set the dataset, cache and person-encoder paths in `configs/proposed.yaml`.
Frozen assets are configured in `configs/person_encoder.yaml`.

| Config key | Default path | Contents |
| --- | --- | --- |
| `data.final_dir` | `dataset/data/final` | Annotations and fixed splits |
| `data.image_root` | `dataset/data/raw/images` | Machine-local image tree |
| `data.detection_cache` | `cache/detections` | Ordered pixel boxes and image sizes |
| `data.scene_cache` | `cache/scene` | CLIP patch tokens and native global embeddings |
| `data.fafa_image_cache` | `cache/person-images` | Person features and native target tokens |
| `data.fafa_query_cache` | `cache/person-queries` | Text-conditioned query crop features |
| `cache.supervision_cache` | `cache/train-supervision.json` | Train-only head alignment |

Extraction resumes incomplete builds and reuses completed caches. The four
feature-cache roots must be separate; visual features contain no training labels.
Native query caches must cover train and validation before training starts.
Completed caches support training and retrieval without loading FAFA weights.
See [cache preparation and reuse](docs/detection_cache.md).

## Run experiments

Run from the repository root. The validation recipe prepares assets, builds or
reuses caches, and starts a fresh training run with validation calibration:

```bash
bash scripts/methods/proposed.bash val
bash scripts/methods/proposed.bash test  # After validation selection completes.
```

To run stages individually:

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py run --config runs/proposed/selected.yaml --split test
```

`run` retrieves and evaluates an existing checkpoint. `retrieve` and `evaluate`
run those stages separately. `prepare-supervision` prepares train-only alignment;
`build-cache --cache-stage` selects `detections`, `scene`, `fafa-image`,
`fafa-query`, or both FAFA stages (`fafa`).

`--set key=value ...` applies typed overrides to existing YAML keys. Use a fresh
output directory for each experiment:

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set train.seed=1 output.dir=runs/proposed-seed1
```

Continue an interrupted run using its saved effective configuration:

```bash
.venv-proposed/bin/python tools/run.py train --config runs/proposed/config.yaml --resume
```

`--resume runs/proposed/last.pt` selects that checkpoint explicitly. Keep the
complete run directory, original source, inputs, software runtime and numerical
settings, including the total epoch count. Resume restores AdamW, AMP scaler,
RNGs, batch position, sampler streams, hard pools and calibrated best selection.
`last.pt` is saved periodically, before validation and after each completed epoch.
SIGINT/SIGTERM request a save after the current batch; a forced kill returns to
the latest durable checkpoint. `best.pt` is a compact inference artifact and
cannot resume training. See the [resume contract](docs/proposed_runs.md#resume-training).

Training publishes `selected.yaml` and `selected/val/` whenever calibrated
validation improves. Test uses this selection directly. Optional offline
calibration compares exported validation artifacts:

```bash
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
```

W&B defaults to online tracking. Configure authentication before training, or
set `wandb.mode=offline` or `wandb.enabled=false`.

| Default output | Contents |
| --- | --- |
| `runs/proposed/` | Effective config, history, checkpoints, `selected.yaml` and selection summary |
| `runs/proposed/evaluation/epoch_NNN/` | Calibrated metrics and every grid candidate |
| `runs/proposed/selected/{val,test}/` | `rankings.pt`, `run.json`, `metrics.json` |
| `runs/proposed-calibration/` | Optional offline calibration results |

Score artifacts support [CPU replay](docs/proposed_runs.md#score-artifacts-and-cpu-replay)
without encoders or feature caches. Use the [reporting guide](docs/experiments.md)
to export saved metrics.

## Dataset and reproducibility

Dataset version `0.1.0` contains 8,999 queries and 37,107 images. Each query searches
its original PIPA split, excluding the reference image. Test never trains or
selects parameters. See the [benchmark protocol](docs/benchmark_protocol.md),
[dataset construction](dataset/README.md) and [annotation UI](labelstudio/README.md).

`train.seed` seeds Python, NumPy and PyTorch, with independent named streams for
epoch sampling. `runtime.deterministic=true` requests reproducible kernels.
Record the commit, effective YAML, seed, cache IDs, software stack and GPU model;
see [reproducibility details](docs/proposed_runs.md#seeds-and-reproducibility).

## Development

`src/rcr/proposed/` owns the model, training, caches and ranking.
`src/rcr/common/` owns shared data, configuration and I/O;
`src/rcr/dataset/` owns annotation construction and audits;
`src/rcr/evaluation/` owns metrics and reports.
The [script reference](scripts/README.md) documents the CLI workflows.

```bash
bash scripts/setup.bash dev
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/python -m ruff check src tools tests labelstudio
.venv-dev/bin/python -m ruff format --check src tools tests labelstudio
```

The CPU suite uses small fixtures without pretrained downloads. It covers
training/backward, interrupted resume, cache contracts, hard mining, retrieval,
calibration and frozen test selection.
