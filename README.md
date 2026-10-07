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

- `scripts/run.py`: the single experiment CLI.
- `configs/{proposed,clip,fafa}.yaml`: one config per method.
- `configs/ablations/*.yaml`: explicit experiments.
- `src/rcr/proposed/`: training, losses, ranking and experiments; `nn/` holds
  model components and `cache/` holds frozen feature extraction/loading.
- `src/rcr/baselines/`: CLIP, official FAFA adapter and val tuning.
- `src/rcr/common/`: shared data loading, runtime, I/O and detection utilities.
- `src/rcr/evaluation/`: shared benchmark protocol and metrics.
- `dataset/`, `src/rcr/dataset/`, `scripts/data/`, `labelstudio/`: dataset pipeline.
- `scripts/`: experiment CLI, isolated FAFA worker and dataset launchers.

See [where to edit and the path migration table](docs/code_structure.md).

Dependency direction: CLI → method functions → model / shared utilities.
Model and loss modules do not import the CLI. There is no trainer framework,
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

Use Python 3.11/3.12. Run commands from the repository root. Keep proposed and
native FAFA in separate environments because they need different Transformers:

```bash
python3.11 -m venv .venv-proposed
.venv-proposed/bin/python -m pip install -r requirements/bootstrap.txt
.venv-proposed/bin/python -m pip install --no-build-isolation -r requirements/proposed.txt
python3.11 -m venv .venv-fafa
.venv-fafa/bin/python -m pip install -r requirements/bootstrap.txt
.venv-fafa/bin/python -m pip install --no-build-isolation -r requirements/fafa.txt
```

Set `data.dino_cache`, `data.cache`, raw-image paths and `person_encoder.python`
in the proposed YAML. DINO scenes/detections and FAFA person features live in
separate directories. `build-cache --cache-stage dino` builds/reuses the DINO
source; `--cache-stage persons` builds/reuses FAFA only. The default runs both.
Completed compatible caches are skipped before model/worker loading; interrupted
builds resume saved images. Training/retrieval join both caches without FAFA.
See [Kaggle reuse and commands](docs/proposed_runs.md).

```bash
# Main model: build/reuse caches, train, validation.
.venv-proposed/bin/python scripts/run.py run --config configs/proposed.yaml --prepare --build-cache --train --splits val

# Joint 2D validation calibration; only the selected pair reaches test.
.venv-proposed/bin/python scripts/run.py ablate --config configs/calibration.yaml --splits val test

# Score-term removals from the same calibrated checkpoint/shortlist.
.venv-proposed/bin/python scripts/run.py ablate --config configs/ablations/binding.yaml --splits val test

# CLIP and native FAFA baselines use their respective environments/configs.
.venv-clip/bin/python scripts/run.py run --config configs/clip.yaml --prepare
.venv-fafa/bin/python scripts/run.py run --config configs/fafa.yaml --prepare
```

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
.venv-proposed/bin/python -m pip install -e '.[dev]'
.venv-proposed/bin/python -m pytest -q
python -m ruff check src scripts tests labelstudio
```

Tests use tiny local encoders/caches, including gradient isolation, same-person
binding, separate scene/person widths, empty sets, padding and the train/retrieve
pipeline. The FAFA extraction contract is tested with a fake native trunk; tests
do not download full weights or claim PIPA quality/CUDA throughput. Optional
CLIP integration and CUDA tests require their corresponding dependencies/device.

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
sequence is:

```bash
bash scripts/data/phase1_rewrite.bash prepare
python scripts/data/prepare_handoffs.py positives
bash scripts/data/phase2_finalize.bash --version 0.2.0
```

`phase2_finalize.bash` validates positive decisions and writes the deterministic
final export. It does not call Gemini for already accepted Stage-2 samples.
