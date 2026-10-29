# Referential Composition Retrieval

Research code and benchmark for **Referential Composition Retrieval (RCR)**.
Each query contains a reference image, text identifying one or more Subjects,
and a requested change. Retrieval ranks images that preserve those identities
and satisfy the change.

The code uses plain PyTorch, YAML configurations and explicit experiment stages:

```text
final dataset -> frozen feature caches -> train -> retrieve -> evaluate
```

## Method

Frozen CLIP ViT-B/32 encodes scenes and text; frozen FAFA encodes person crops.
The learned RCR modules:

1. Ground Subject descriptions to detected people in the reference image.
2. Retrieve a coarse shortlist using person identity and global state scores.
3. Compute one partial, one-to-one identity assignment with a null option.
4. Rerank the shortlist using native FAFA composition and relational context.

Training and inference retain their separate objectives and scoring weights:

```text
L = L_ground + 0.1 L_id + L_coarse + L_context + L_state
S_fine = alpha * B_FAFA + context_weight * z_topM(coverage * context)
S_final = z_topM(S_fine) + fine_coarse_weight * z_topM(S_coarse)
```

New runs fix `alpha=1`; compatible checkpoints retain their recorded scale.
Fine normalization uses the complete coarse Top-M, FP32 population standard
deviation and epsilon `1e-6`. A zero fusion weight returns the fine score directly.
The remaining gallery retains coarse order.

The current YAML runs 35 epochs, with context disabled for the first 6 and
cosine learning-rate decay over the full run. Hard-negative mining starts at
epoch 11, independently of context warmup.
Grounding, identity, coarse and state objectives remain active during warmup.
Validation selects `best.pt` by the mean of overall and macro-case Full-mAP;
offline calibration then selects scoring parameters using validation Full-mAP.
Test uses the frozen selection.

See [method and equations](docs/proposed_method.md),
[feature contracts](docs/feature_contracts.md) and
[experiment protocol](docs/proposed_runs.md) for the full definitions.

## Setup

Use Python 3.11 or 3.12. Proposed requires PyTorch >=2.6 for the pinned CLIP
checkpoint; native FAFA uses PyTorch 2.5.1. Install them in separate environments:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash fafa
```

Without `PYTHON`, setup searches for Python 3.11, then 3.12, then a compatible
`python3`. An explicit `PYTHON` must be compatible; Python 3.13 is unsupported
by the pinned NumPy/FAFA stack. Setup checks this before creating an environment.
It also isolates installation from notebook `PYTHONPATH` and user startup hooks.

On a Kaggle image that only has Python 3.13, install Python 3.11 and seed pip
with uv before setup:

```bash
python3 -m pip install --no-cache-dir uv
uv python install 3.11
# If a previous setup failed, move that .venv-proposed/.venv-fafa aside first.
for method in proposed fafa; do
  uv venv --python 3.11 --seed ".venv-$method"
  PYTHON=".venv-$method/bin/python" bash scripts/setup.bash "$method"
done
```

Setup can be rerun after an interrupted package install. Always rerun it to
repair an incomplete environment: the existence of `bin/python` alone does not
mean dependencies are installed. An existing environment with an incompatible
Python version must be moved aside before setup; caches and runs are separate.

Edit the paths in `configs/proposed.yaml`:

```yaml
data:
  final_dir: dataset/data/final
  image_root: dataset/data/raw/images
  detection_cache: cache/detections
  scene_cache: cache/scene
  fafa_image_cache: cache/person-images
  fafa_query_cache: cache/person-queries
person_encoder:
  python: .venv-fafa/bin/python
```

Raw images are not committed. Prepare them using the
[dataset guide](dataset/README.md). The separate FAFA environment is needed for
asset preparation and person-feature extraction; completed caches support
training and retrieval using only the proposed environment.

## Run experiments

The validation recipe prepares assets, builds or reuses caches, trains,
retrieves validation scores, and calibrates them offline:

```bash
bash scripts/methods/proposed.bash val
bash scripts/methods/proposed.bash test  # After validation selection completes.
```

For an individual stage, run from the repository root:

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py run --config runs/proposed/config.yaml --split val
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
.venv-proposed/bin/python tools/run.py run --config runs/proposed-calibration/selected.yaml --split test
```

`run` retrieves and evaluates one split from prepared caches and an existing
checkpoint. It does not train. Use `retrieve` or `evaluate` to run either stage
alone, and `prepare-supervision` to prepare train-only head alignment explicitly.
Use `build-cache --cache-stage` for `detections`, `scene`, `fafa-image`,
`fafa-query`, or both FAFA stages (`fafa`).

Configuration choices live in YAML. `--set key=value ...` provides typed
overrides for existing keys; `--split` overrides the configured split:

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set train.seed=1 output.dir=runs/proposed-seed1
.venv-proposed/bin/python tools/run.py run --config runs/proposed-seed1/config.yaml --split val
```

Use a fresh output directory for each training run. Training refuses an existing
run; retrieval, evaluation, calibration and reports replace their active outputs.
When using another run, point `configs/calibration.yaml: rankings` to its saved
`val/rankings.pt`. The Bash recipe derives the run/selection paths from YAML and
passes the active run's rankings to calibration. W&B tracking defaults to online;
set `wandb.mode=offline` or `wandb.enabled=false` as needed.

CLIP and FAFA baselines have separate environments and recipes. See the
[baseline guide](docs/baselines.md) and [script reference](scripts/README.md).

## Caches and outputs

| Config key | Cached evidence | Rebuild when |
| --- | --- | --- |
| `data.detection_cache` | Ordered pixel boxes and original image sizes | Detector or image inputs change |
| `data.scene_cache` | CLIP patch tokens and native global embeddings | Scene encoder or preprocessing changes |
| `data.fafa_image_cache` | Pooled person features and native target tokens | FAFA image extraction or detections change |
| `data.fafa_query_cache` | Text-conditioned query crop features | Query text, FAFA extraction or detections change |

Extraction is resumable and completed caches are reused. Visual caches contain
no training labels; train-only alignment is stored separately under
`cache.supervision_cache`. Inference receives only images, text and public
Subject slots. See [cache compatibility and detection inputs](docs/detection_cache.md).
Cache paths follow their role: `cache/detections`, `cache/scene`,
`cache/person-images`, `cache/person-queries`, and `cache/train-supervision.json`.

| Default path | Contents |
| --- | --- |
| `runs/proposed/` | Effective config, runtime, history, tokenizer, `last.pt`, `best.pt` |
| `runs/proposed/val/` | `rankings.pt`, `run.json`, `metrics.json` |
| `runs/proposed-calibration/` | Grid summary, `selection.json`, `selected.yaml` |
| `runs/proposed-calibration/selected/{val,test}/` | Selected split results |

The default method, W&B run name and training folder use `proposed`.
`proposed-calibration` names the separate scoring-selection stage. Reports read
its `selected/<split>` results by default; use `--run proposed=...` for another
experiment. Ablation configs retain their own experiment names. Existing runs
keep their saved paths; use their saved configs without renaming their folders.

Validation saves complete raw score matrices for offline calibration/replay.
It fine-scores every non-self query/gallery pair; periodic training validation
and test fine-score only Top-M. Raw matrices require about `20*Q*G` bytes in
addition to rankings. Checkpoints omit optimizer state and frozen backbone
weights, so they are inference artifacts rather than resumable training states.
See [score artifacts and CPU replay](docs/proposed_runs.md#score-artifacts-and-cpu-replay).

## Dataset

Version `0.1.0` contains 4,311 queries and 37,107 images. Each query searches its
own PIPA split, excluding the reference image.

| Split | INDIVIDUAL | GROUP | DUAL | RELATIONAL | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,472 | 137 | 269 | 155 | 4,033 |
| val | 194 | 12 | 36 | 22 | 264 |
| test | 5 | 2 | 3 | 4 | 14 |

The 14-query test split checks the pipeline; it is not a publication-scale
holdout. See the [benchmark protocol](docs/benchmark_protocol.md),
[dataset construction](dataset/README.md) and [reporting guide](docs/experiments.md).

## Code layout and development

| Path | Responsibility |
| --- | --- |
| `configs/` | Experiment and calibration settings |
| `src/rcr/common/` | Data boundaries, configuration, I/O and runtime helpers |
| `src/rcr/dataset/` | Dataset construction and annotation audits |
| `src/rcr/baselines/` | CLIP and native FAFA baselines |
| `src/rcr/proposed/cache/` | Frozen feature extraction and cache readers |
| `src/rcr/proposed/nn/` | Model equations and learned modules |
| `src/rcr/proposed/{train,ranking,calibration,replay}.py` | Optimization, inference and offline selection |
| `src/rcr/evaluation/` | Metrics, provenance checks and reports |
| `tools/`, `scripts/`, `tests/` | CLI, reproducible workflows and CPU contracts |

```bash
bash scripts/setup.bash dev
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/python -m ruff check src tools tests labelstudio
.venv-dev/bin/python -m ruff format --check src tools tests labelstudio
```

For ablations, change one component, keep data/evaluation fixed, and record the
commit, effective YAML, seed and runtime. Model changes belong in the module
that owns the equation; the CLI and Bash recipes only sequence stages.
