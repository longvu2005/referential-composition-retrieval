# Referential Composition Retrieval (RCR)

This repository contains the dataset pipeline, proposed retrieval model, and
shared evaluation protocol for **Referential Composition Retrieval (RCR)**.

## Task

Given:

1. a query image;
2. referential text that identifies one or more Subjects in that image; and
3. a target change/condition;

the system ranks gallery images that preserve the referenced identities and
satisfy the requested target condition.

A **Subject** is a semantic reference and may contain one or more identities.
The benchmark uses four case types:

- `INDIVIDUAL`: one Subject with exactly one identity;
- `GROUP`: one Subject with two or more identities;
- `DUAL`: two Subjects with independent Subject-specific changes;
- `RELATIONAL`: two Subjects connected by an ordered relation.

## Code layout

- `tools/run.py`: the single experiment CLI.
- `tools/report.py`: validate saved JSON and export overall/by-case paper tables.
- `configs/{proposed,clip,fafa}.yaml`: one config per method.
- `configs/ablations/*.yaml`: explicit experiments.
- `src/rcr/proposed/`: training, losses, ranking and experiments; `nn/` holds
  model components and `cache/` holds frozen feature extraction/loading.
- `src/rcr/baselines/`: CLIP, official FAFA adapter and val tuning.
- `src/rcr/common/`: shared data loading, runtime, I/O and detection utilities.
- `src/rcr/evaluation/`: shared benchmark protocol and metrics.
- `dataset/`, `src/rcr/dataset/`, `tools/data/`, `labelstudio/`: dataset pipeline.
- `tools/`: Python commands for individual stages, reports and the FAFA worker.
- `scripts/`: Bash workflows that sequence those commands for reproduction.
- `requirements/`: separate installations for methods, annotation and evaluation.

See [where to edit and module responsibilities](docs/code_structure.md).

Dependency direction: CLI → method functions → model / shared utilities.
YAML parsing lives in `common/config.py`; saved-result evaluation lives in
`evaluation/runner.py`. Model and loss modules do not import the CLI. There is no trainer framework,
recursive config inheritance or plugin registry. An ablation suite reads one
method YAML and applies flat overrides. Method dependencies are imported only
when that method/stage is used.

## Architecture

The proposed model uses frozen **DINO scene patches** and a frozen **FAFA
image-only person trunk**. Mean hidden Q-Former tokens produce each cached crop
feature. Separate trainable heads produce normalized identity (128) and semantic
(384) embeddings. Identity is trained only by supervised contrastive loss;
semantic features feed selection grounding and instruction composition.

Fine scoring combines identity and composed semantics on the SAME target person,
then pools soft Subject memberships. Its learned positive scale adds that binding
score to the existing context/set reasoner. Scene/geometry reasoning handles
conditions outside crops. Global-state coarse scoring and the official evaluation
protocol remain. See [equations and limitations](docs/proposed_method.md).

## Setup and run

Use Python 3.11/3.12. Each environment installs only its own requirements file.
Dependency versions live in `pyproject.toml`; the requirements files select one
extra without duplicating version lists. `requirements.txt` installs the shared
package only. Keep proposed and native FAFA separate because their Transformers
versions conflict:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash fafa
# Only needed when running CLIP baselines:
PYTHON=python3.11 bash scripts/setup.bash clip
```

The setup workflow creates `.venv-<name>`, bootstraps pip from the host (which
needs pip 22.3+), then installs `requirements/<name>.txt`. It supports notebook
images without `ensurepip`. Run standalone `tools/*.py` commands from the repo
root; Bash workflows locate that root themselves. Set `PYTHON` to an existing
environment's Python to use it with a workflow instead of the default venv.

| Install | Purpose |
| --- | --- |
| `requirements/proposed.txt` | Proposed training, DINO cache and retrieval |
| `requirements/clip.txt` | All four OpenAI CLIP variants |
| `requirements/fafa.txt` | Native FAFA baseline or proposed person-feature worker |
| `requirements/dataset.txt` | Annotation preparation and local UIs; no model libraries |
| `requirements/dataset-clip.txt` | Optional OpenCLIP ordering of new positive tasks |
| `requirements/evaluation.txt` | Evaluate saved `.pt` rankings without encoder libraries |
| `requirements/dev.txt` | CPU tests and linting |

Set `data.dino_cache`, `data.cache`, raw-image paths and `person_encoder.python`
in the proposed YAML. DINO scenes/detections and FAFA person features live in
separate directories. `build-cache --cache-stage dino` builds/reuses the DINO
source; `--cache-stage persons` builds/reuses FAFA only. The default runs both.
Completed compatible caches are skipped before model/worker loading; interrupted
builds resume saved images. Training/retrieval join both caches without FAFA.
See [Kaggle reuse and commands](docs/proposed_runs.md).

```bash
# Main model: prepare, build/reuse caches, train, then calibrate on validation.
bash scripts/methods/proposed.bash val

# Joint 2D validation calibration; only the selected pair reaches test.
.venv-proposed/bin/python tools/run.py ablate --config configs/calibration.yaml --splits val test

# Score-term removals from the same calibrated checkpoint/shortlist.
.venv-proposed/bin/python tools/run.py ablate --config configs/ablations/binding.yaml --splits val test

# CLIP and native FAFA: prepare assets, retrieve and evaluate validation.
bash scripts/methods/clip.bash val
bash scripts/methods/fafa.bash val
.venv-proposed/bin/python tools/report.py --split val \
  --run proposed=runs/calibration/selected/val
```

For a frozen test run, use `bash scripts/methods/<method>.bash test`. Proposed
loads `runs/calibration/selected.yaml`; CLIP/FAFA reuse their validation selections.
The test workflows never train or select weights. For a custom config, stage or
override, call `tools/run.py` directly. See [workflow contracts](scripts/README.md).

The existing Kaggle DINO cache can be used read-only as `data.dino_cache`; only
FAFA person features need extraction. Legacy files require the explicit
`cache.allow_legacy_dino` setting and matching original preprocessing. Train new
heads for the new architecture. Cache/source IDs, gallery order, encoder metadata
and checkpoint identity are checked. Keep historical run directories separately.

| Command | Purpose |
| --- | --- |
| `prepare` | Prepare proposed person encoder or baseline assets |
| `build-cache` | Build/reuse DINO and/or FAFA caches (`--cache-stage`) |
| `train` | Train from cache; validation selects best.pt |
| `retrieve` | Save complete split-gallery rankings |
| `evaluate` | Evaluate saved rankings without a model |
| `run` | Coordinate prepare/build/train/retrieve/evaluate |
| `ablate` | Explicit training/inference suites or validation calibration |

`--set key=value ...` overrides existing YAML fields. `--splits` selects splits.
`--max-queries` is only for matching retrieve/evaluate smoke runs. Normal run and
ablation use complete requested query splits. Outputs contain checkpoints,
config/tokenizer, history, per-epoch metrics, rankings, run metadata and summaries.
W&B retains its metrics and additionally logs the learned binding scale.

Detailed [proposed workflow](docs/proposed_runs.md),
[baseline workflow](docs/baselines.md), and [new ablations](docs/ablations.md).
Ablations now cover DINO/FAFA x shared/dual and binding term removals, with optional
retraining. The old identity-amplitude, coarse-only, loss and sampling suites
have been removed. Calibration lives separately in `configs/calibration.yaml`.

## Tests

```bash
.venv-proposed/bin/python -m pip install -r requirements/dev.txt
.venv-proposed/bin/python -m pytest -q
.venv-proposed/bin/python -m ruff check src tools tests labelstudio
```

The CPU contract suite also runs in GitHub Actions on Python 3.11 and 3.12.
`dev` includes SciPy for the FAFA matching adapter tests; native FAFA weights
are not needed.

Tests use tiny local encoders/caches, including gradient isolation, same-person
binding, separate scene/person widths, empty sets, padding and the train/retrieve
pipeline. The FAFA extraction contract is tested with a fake native trunk; tests
do not download full weights or claim PIPA quality/CUDA throughput. Optional
CLIP integration and CUDA tests require their corresponding dependencies/device.

For an isolated CLIP environment, see [baseline setup](docs/baselines.md).
The deterministic dataset pipeline uses `requirements/dataset.txt`; optional
candidate ordering uses `requirements/dataset-clip.txt`. Report generation reads
JSON and works with the base package; decoding saved rankings needs the
evaluation environment. No model client is required for annotation preparation.

## Current dataset snapshot

The checked-in final export is **version 0.1.0** with 4,311 queries and
37,107 registered gallery images.

Each query searches **all images in its own PIPA split**, excluding the query
image: 17,000 train images, 5,684 val images, or 7,868 test images. The 6,555
leftover images remain in the image registry/cache but never enter a query's
gallery. Gallery membership comes from `images.jsonl` paths, not the images
appearing in annotated query/target pairs. Full Positives and identity-positive
metrics use the same split gallery.

Stored annotations and final data stay unchanged. The loader intersects reviewed
`positive_image_ids` with the query split in memory, leaving the original labels
available for annotation provenance and other retrieval protocols.

| Split | INDIVIDUAL | GROUP | DUAL | RELATIONAL | Total |
|---|---:|---:|---:|---:|---:|
| train | 3,472 | 137 | 269 | 155 | 4,033 |
| val | 194 | 12 | 36 | 22 | 264 |
| test | 5 | 2 | 3 | 4 | 14 |
| **all** | **3,671** | **151** | **308** | **181** | **4,311** |

The current test split is intentionally very small and should be treated as a
pipeline-validation split, not yet as the final publication-scale test set.

Final benchmark files live in `dataset/data/final/`:

```text
samples.jsonl
images.jsonl
gallery.jsonl
head_boxes.jsonl
manifest.json
splits/{train,val,test}.txt
```

Each final sample contains:

```text
sample_id
case_type
query_image_id
target_image_id
positive_image_ids
subjects[{subject_id, identity_ids[]}]
final_desc
final_change
final_instruction
```

The local raw image collection is not part of the repository archive and must
be available under the configured `image_root`.

### Dataset reconstruction

The accepted Stage-2 text is treated as immutable input. The normal rebuild
sequence is below. The source export currently contains 8,001 tasks. Review and
positive completion counts are reported by the preparation tools. A full rebuild
requires every selected task's labels; use `--allow-partial` during annotation.
The checked-in final snapshot remains unchanged.

```bash
bash scripts/setup.bash dataset
bash scripts/data/prepare_review.bash
# Complete or correct human reviews, then prepare positive tasks:
bash scripts/data/prepare_positives.bash
# Complete positive decisions, stop the UI, then export:
bash scripts/data/finalize.bash --version 0.2.0
```

Review preparation attaches image/Subject context directly to accepted Stage 2
text. It preserves completed reviews and never creates completed labels for new
tasks. `finalize.bash` validates positive decisions and writes the deterministic
final export. See the [dataset workflow](dataset/README.md).
