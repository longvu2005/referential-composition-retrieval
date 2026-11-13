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

Training and inference use the same fine ranking score:

```text
L = L_ground + 0.1 L_id + L_coarse + L_context + L_state
h = tanh(context)
S_fine = alpha * B_FAFA + beta * (coverage * h - (1 - coverage))
S_final = S_fine
```

New runs fix `alpha=1`; `beta` is `retrieval.context_weight`, shared with training.
The residual lies in `[-beta, beta]`. Holding binding and raw context fixed,
reducing coverage cannot increase the score. There is no shortlist-dependent
fine normalization or final coarse fusion. Coarse scores select Top-M and order
the remaining gallery. The checked-in beta is a starting value, not a measured
optimum.

`train.epochs` sets the total run length; `train.context_warmup_epochs` disables
context initially while grounding, identity, coarse and state stay active.
Both optimizer groups use cosine decay throughout the run.
After the sampling warmup, train-only fine mining selects hard same-ID condition
negatives and wrong/missing-ID negatives. Context learns from semantic pairs plus
a rejection term weighted by `loss.context_rejection_weight`; the checked-in YAML
sets `semantic_hard_fraction=0.1`, `hard_fraction=0.1` and
`context_rejection_weight=0.2`. Omitted semantic-hard and rejection settings
default to zero. Fine mining reads frozen FAFA query
tokens for train; ensure the query cache covers train before starting.

Context learns a residual on frozen native FDA binding, using the same
`retrieval.pooling`, assignment and score coefficient as inference. Its pairwise
loss uses `loss.context_temperature`; `loss.context_weight` weights this task
in the total loss. Native FAFA query/target tokens are read during context
training, so the query cache must cover train. Warmup skips these reads.
`context_fda_error` logs the fraction of eligible pairs that FDA ranks wrong or
tied. See the [residual objective](docs/proposed_method.md#residual-context-objective).

Fine scoring requires a newly trained checkpoint. Previous scoring modes,
residual-scale configs and schema-1 replay artifacts are rejected. Existing
frozen feature caches remain reusable when their provenance matches.
Validation selects `best.pt` by the mean of overall and macro-case Full-mAP;
offline calibration uses the same criterion by default and includes the source
configuration as a baseline (`selection_metric: full_map` selects overall-only).
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
`val/rankings.pt`. W&B tracking defaults to online; set `wandb.mode=offline` or
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
| Candidate sampling | `positives_per_query=2`, `identity_fraction=0.5`, `hard_fraction=0.1`, `semantic_hard_fraction=0.1` |
| Fine mining | `warmup_epochs=8`, `refresh_every_epochs=2`, `pool_size=100` |
| Identity crops | `max_crops=128`, `crops_per_identity=2`, `random_identities=8`, `hard_negative_fraction=0.1` |
| Loss | weights `1`/`0.1`/`1`/`1`/`1` for grounding/identity/coarse/context/state, `context_rejection_weight=0.2`, temperatures `0.1`, including `context_temperature` |
| Model | `dim=384`, `identity_dim=128`, `state_dim=512`, `num_heads=6`, `mlp_ratio=2`, `geo_dim=32`, `dropout=0.1`, `fda_k=6`, `coarse_beta=0.4` |
| Matching | `background_logit=-3.5`, `selection_threshold=0.5`, `identity_threshold=0.2`, `missing_penalty=1.1`, `duplicate_iou=0.9` |
| Retrieval | `pooling=subject_mean`, `fine_score_mode=bounded_residual`, `context_weight=0.1`, `coarse_beta=1.16`, `top_m=500`, `candidate_ks=[500]` |
| Validation | `evaluation.every_epochs=4` |

With two positives and enough same-ID evidence, the sampling rows give each query 2
positives, 2 hard and 13 random same-ID negatives, 3 hard wrong-ID negatives and 12
uniform negatives. `retrieval.coarse_beta` overrides `model.coarse_beta` at
inference. The residual weight and pooling are fixed from training;
`calibrate` selects coarse beta and Top-M for test and writes the complete
configuration to `runs/proposed-calibration/selected.yaml`.

CLIP and FAFA baselines have separate environments and recipes. See the
[baseline guide](docs/baselines.md) and [script reference](scripts/README.md).

`fafa_pooling` compares two frozen FAFA aggregation rules in one run, reusing
FAFA's environment and feature caches; see [FAFA pooling](docs/baselines/fafa_pooling.md).

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

Validation saves complete raw score matrices for offline calibration/replay.
It fine-scores every non-self query/gallery pair; periodic training validation
and test fine-score only Top-M. Raw matrices require about `28*Q*G` bytes in
addition to rankings; `fafa_binding` shares the selected pooling matrix.
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
| `configs/` | Experiment and calibration settings |
| `src/rcr/common/` | Data boundaries, configuration, I/O and runtime helpers |
| `src/rcr/dataset/` | Dataset construction and annotation audits |
| `src/rcr/baselines/` | CLIP and native FAFA baselines |
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

The CPU suite covers feature-cache construction, residual scoring and coverage,
training/backward, hard mining, retrieval, validation selection and frozen test
inference. Shared fixtures live in `tests/conftest.py`; small cached inputs live
in `tests/helpers.py`.

For ablations, change one component, keep data/evaluation fixed, and record the
commit, effective YAML, seed and runtime. Model changes belong in the module
that owns the equation; the CLI and Bash recipes only sequence stages.
