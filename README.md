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

- `tools/methods/run.py`: the single experiment CLI.
- `configs/methods/{proposed,clip,fafa}.yaml`: one config per method.
- `configs/ablations/*.yaml`: explicit experiments.
- `src/rcr/methods/proposed/`: cache, train loop, model, loss and retrieval.
- `src/rcr/methods/baselines/`: CLIP, official FAFA adapter and val tuning.
- `src/rcr/methods/common/`: shared loader, device, result paths and evaluation I/O.
- `src/rcr/evaluation/`: shared benchmark protocol and metrics.
- `dataset/`, `src/rcr/dataset/`, `tools/dataset/`, `labelstudio/`: dataset pipeline.
- `scripts/`: dataset launchers; method experiments use the Python CLI directly.

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

Set data/cache paths and `person_encoder.python` in the proposed YAML. `prepare`
downloads only FAFA model assets for proposed. Cache construction runs detector
and DINO first, releases their GPU storage, then invokes FAFA in its own venv.
Training and retrieval read cached features without loading FAFA.

```bash
# Main model: new cache, train, validation.
.venv-proposed/bin/python tools/methods/run.py run --config configs/methods/proposed.yaml --prepare --build-cache --train --splits val

# Joint 2D validation calibration; only the selected pair reaches test.
.venv-proposed/bin/python tools/methods/run.py ablate --config configs/calibration/joint.yaml --splits val test

# Score-term removals from the same calibrated checkpoint/shortlist.
.venv-proposed/bin/python tools/methods/run.py ablate --config configs/ablations/binding.yaml --splits val test

# CLIP and native FAFA baselines use their respective environments/configs.
.venv-clip/bin/python tools/methods/run.py run --config configs/methods/clip.yaml --prepare
.venv-fafa/bin/python tools/methods/run.py run --config configs/methods/fafa.yaml --prepare
```

Old DINO caches/checkpoints require a new build/train for this architecture.
Cache version, encoder/preprocessing metadata and checkpoint cache identity are
checked explicitly. Existing run directories can be kept for historical results.

| Command | Purpose |
| --- | --- |
| `prepare` | Prepare proposed person encoder or baseline assets |
| `build-cache` | Build frozen scene/person features |
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
[baseline workflow](docs/baselines.md), and [new ablations](docs/ablations_vi.md).
Ablations now cover DINO/FAFA x shared/dual and binding term removals, with optional
retraining. The old identity-amplitude, coarse-only, loss and sampling suites
have been removed. Calibration lives separately in `configs/calibration/`.

## Tests

```bash
.venv-proposed/bin/python -m pip install -e '.[dev]'
.venv-proposed/bin/python -m pytest -q
ruff check src tools tests labelstudio
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
bash scripts/phase1_rewrite.bash prepare
python tools/dataset/prepare_handoffs.py positives
bash scripts/phase2_finalize.bash --version 0.2.0
```

`phase2_finalize.bash` validates positive decisions and writes the deterministic
final export. It does not call Gemini for already accepted Stage-2 samples.
