# Referential Composition Retrieval

Proposed method and benchmark for **Referential Composition Retrieval (RCR)**.
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

Training and inference share the same pairwise fine score. Validation and
retrieval fuse that score with coarse evidence over the complete Top-M shortlist:

```text
L = L_ground + 0.1 L_id + L_coarse + L_context + L_state
S_fine = alpha * B_FAFA + a * coverage + b * context
S_final = Z_M(S_fine) + eta * Z_M(S_coarse)
```

`alpha=1` is fixed. `a=retrieval.coverage_weight=0.04` and
`b=retrieval.context_weight=0.10` are fixed during training and inference.
Context is the raw reasoner output, without activation or a coverage gate.
`eta=retrieval.fine_coarse_weight=0.15` is the starting fusion weight.
For positive eta, both branches use population z-scores on the same finite
coarse support within each query's Top-M, never within inference chunks.
A constant branch contributes zero. Eta zero returns raw fine scores.
Stable ties retain coarse order and the gallery tail stays unchanged.
Coefficients are starting values, not measured optima for the new run.

The default run trains for 20 epochs. Context is disabled in epochs 1-4 while
grounding, identity, coarse and global state learn. From epoch 5, all objectives
are active and train-only fine mining refreshes every two epochs. Both optimizer
groups use cosine decay throughout the run. With sufficient candidate pools,
`semantic_hard_fraction=0.3` gives five same-ID hard condition negatives per
32-candidate query. Wrong/missing-ID hard negatives use `hard_fraction=0.1`;
`loss.context_rejection_weight=0.2` weights their rejection objective.

Context ranks the Linear fine score on frozen native FDA binding. Its loss uses
`loss.context_temperature`; `loss.context_weight` weights the task in the total
loss. The coarse and state branches retain their separate objectives; fusion is
postprocessing, also used for validation checkpoint selection and fine mining.
Native FAFA query/target caches must cover train before training starts.
`context_fda_error` records FDA pair errors; see the
[context objective](docs/proposed_method.md#context-ranking-objective).

Use a fresh training run: checkpoints trained with previous fine scorers and
score artifacts from older schemas are rejected. Compatible frozen caches
remain reusable. `best.pt` maximizes the mean of overall and macro-case
validation Full-mAP **after fusion**. Calibration searches a small eta grid on
saved validation scores, with pooling and Linear coefficients fixed. Test uses
the frozen selection.

See [method and equations](docs/proposed_method.md),
[feature contracts](docs/feature_contracts.md) and
[experiment protocol](docs/proposed_runs.md) for the full definitions.

## Setup

Use Python 3.11 or 3.12. Proposed requires PyTorch >=2.6 for the pinned CLIP
checkpoint; frozen FAFA extraction uses PyTorch 2.5.1. Install the training and
person encoder runtimes in separate environments:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash person-encoder
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
for environment in proposed person-encoder; do
  uv venv --python 3.11 --seed ".venv-$environment"
  PYTHON=".venv-$environment/bin/python" bash scripts/setup.bash "$environment"
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
  python: .venv-person-encoder/bin/python
  fafa_config: configs/person_encoder.yaml
```

Raw images are not committed. Prepare them using the
[dataset guide](dataset/README.md). The person encoder environment is needed for
asset preparation and person-feature extraction; completed caches support
training and retrieval using only the proposed environment.

## Run experiments

The validation recipe prepares assets, builds or reuses caches, trains,
retrieves validation scores, and freezes the selected validation result:

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
When using another run, list its saved `val/rankings.pt` under
`configs/calibration.yaml: rankings`; several validation artifacts, for example
retrieved with different `retrieval.coarse_beta`, are compared there. W&B tracking defaults to online; set `wandb.mode=offline` or
`wandb.enabled=false` as needed.

`train.seed` fixes every random choice of a training run: it seeds Python, NumPy
and PyTorch once, and each epoch's sampling streams derive from
`(seed, stream, epoch)`, so different seeds never share a stream.
`runtime.deterministic=true` (the default) also asks PyTorch for reproducible
kernels. The same seed, YAML, caches, software stack and GPU model are meant to
give identical weights; see [seeds and reproducibility](docs/proposed_runs.md#seeds-and-reproducibility).

The checked-in `configs/proposed.yaml` sets these defaults:

| Group | Values |
| --- | --- |
| Schedule | `train.epochs=20`, `context_warmup_epochs=4`, `batch_size=24` queries, `candidates=32`, AMP on, `seed=0` |
| Optimizer | AdamW, `lr=1e-4`, `text_lr=1e-5`, `weight_decay=0.01`, cosine decay to `min_lr_ratio=0.1` |
| Candidate sampling | `positives_per_query=2`, `identity_fraction=0.5`, `hard_fraction=0.1`, `semantic_hard_fraction=0.3` |
| Fine mining | `warmup_epochs=4`, `refresh_every_epochs=2`, `pool_size=100` |
| Identity crops | `max_crops=128`, `crops_per_identity=2`, `random_identities=8`, `hard_negative_fraction=0.1` |
| Loss | weights `1`/`0.1`/`1`/`1`/`1` for grounding/identity/coarse/context/state, `context_rejection_weight=0.2`, temperatures `0.1`, including `context_temperature` |
| Model | `dim=384`, `identity_dim=128`, `state_dim=512`, `num_heads=6`, `mlp_ratio=2`, `geo_dim=32`, `dropout=0.1`, `fda_k=6`, `coarse_beta=0.4` |
| Matching | `background_logit=-3.5`, `selection_threshold=0.5`, `identity_threshold=0.2`, `missing_penalty=1.1`, `duplicate_iou=0.9` |
| Retrieval | `pooling=subject_mean`, `fine_score_mode=linear`, `coverage_weight=0.04`, `context_weight=0.10`, `fine_coarse_weight=0.15`, `coarse_beta=1.16`, `top_m=500`, `candidate_ks=[500]` |
| Validation | `evaluation.every_epochs=3` |

With two positives and enough same-ID evidence, the sampling rows give each query 2
positives, 5 hard and 10 random same-ID negatives, 3 hard wrong-ID negatives and 12
uniform negatives. `retrieval.coarse_beta` overrides `model.coarse_beta` at
inference. Linear coefficients and pooling are fixed from training; coarse beta
and Top-M define the shortlist of each retrieval run. `calibrate` evaluates
`fine_coarse_weights: [0.15, 0.10, 0.20, 0.25]` on saved validation artifacts and
writes the complete configuration to
`runs/proposed-calibration/selected.yaml`.

See the [script reference](scripts/README.md) for environment setup, dataset
workflows and report export.

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
| `runs/proposed-calibration/` | Candidate summary, `selection.json`, `selected.yaml` |
| `runs/proposed-calibration/selected/{val,test}/` | Selected split results |

Every split fine-scores only the Top-M shortlist of each query and saves that
evidence as `Q x M` tensors for offline replay. They require about `20*Q*M`
bytes in addition to rankings; `fafa_binding` shares the selected pooling tensor.
Checkpoints omit optimizer state and frozen backbone
weights, so they are inference artifacts rather than resumable training states.
See [score artifacts and CPU replay](docs/proposed_runs.md#score-artifacts-and-cpu-replay).

## Dataset

Version `0.1.0` contains 8,999 queries and 37,107 images. Each query searches its
own PIPA split, excluding the reference image.

| Split | INDIVIDUAL | GROUP | DUAL | RELATIONAL | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,590 | 444 | 395 | 464 | 4,893 |
| val | 1,320 | 168 | 149 | 162 | 1,799 |
| test | 1,319 | 329 | 318 | 341 | 2,307 |

Test never trains or selects parameters. See the
[benchmark protocol](docs/benchmark_protocol.md),
[dataset construction](dataset/README.md) and [reporting guide](docs/experiments.md).

## Code layout and development

| Path | Responsibility |
| --- | --- |
| `configs/` | Proposed experiment, frozen person encoder and calibration settings |
| `src/rcr/common/` | Data boundaries, configuration, I/O and runtime helpers |
| `src/rcr/dataset/` | Dataset construction and annotation audits |
| `src/rcr/proposed/cache/` | Frozen feature extraction and cache readers |
| `src/rcr/proposed/nn/` | Model equations and learned modules |
| `src/rcr/proposed/{train,ranking,calibration,replay}.py` | Optimization, inference and offline selection |
| `src/rcr/evaluation/` | Metrics, provenance checks and reports |
| `tools/`, `scripts/`, `tests/` | CLI, reproducible workflows and CPU workflow tests |

```bash
bash scripts/setup.bash dev
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/python -m ruff check src tools tests labelstudio
.venv-dev/bin/python -m ruff format --check src tools tests labelstudio
```

The CPU suite covers feature-cache construction, Linear scoring, coverage and coarse fusion,
training/backward, hard mining, retrieval, validation selection and frozen test
inference. Shared fixtures live in `tests/conftest.py`; small cached inputs live
in `tests/helpers.py`.

For ablations, change one component, keep data/evaluation fixed, and record the
commit, effective YAML, seed and runtime. Model changes belong in the module
that owns the equation; the CLI and Bash recipes only sequence stages.
