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

The proposed v2 model uses frozen **FAFA identity features**, **CLIP image/text
features from one checkpoint**, and **DINO scene patches**. Its trainable path is
categorical Subject/background grounding → structured full-condition composition
→ identity-guided soft partial transport → bound target evidence → joint reasoning.
Each Subject can retain several members; case labels and GT cardinalities never
enter inference. Final score is `S_id + logsigmoid(condition_logit)`.

Default retrieval fine-scores identity top-500 and appends the coarse tail;
`retrieval.mode=full` is the full-gallery control. There is no global-state score,
z-score fusion or coarse/fine weighted sum. Only overall validation Full-mAP
selects the checkpoint. See [equations and contracts](docs/proposed_method.md).

## Setup and run

Use Python 3.11/3.12 and install the selected method in the notebook's current
Python. Each notebook runs one method. Requirements files select dependency
profiles from `pyproject.toml`; the setup script installs them without creating
virtual environments. Proposed includes the FAFA runtime needed for its person
cache, plus DINOv3/CLIP and training dependencies.

```bash
# Proposed notebook (includes its FAFA person encoder):
bash scripts/setup.bash proposed
# In a CLIP or FAFA baseline notebook, choose clip or fafa instead.
```

In Colab/Kaggle, bind Bash workflows to the kernel's Python before setup:

```python
import os
import sys

os.environ["PYTHON"] = sys.executable
```

Then run `!bash scripts/setup.bash proposed` from the repository root. Install
before importing model packages; restart the kernel if those packages were
already imported. Bash workflows use `PYTHON` when set, otherwise `python` from
PATH. Direct notebook commands can use `{sys.executable}`. Tools never install
packages during cache building, training or retrieval.

| Install | Purpose |
| --- | --- |
| `requirements/proposed.txt` | Complete proposed method: DINO/FAFA/CLIP caches, training and retrieval |
| `requirements/clip.txt` | All four OpenAI CLIP variants |
| `requirements/fafa.txt` | Native FAFA baseline or proposed person-feature worker |
| `requirements/dataset.txt` | Annotation preparation and local UIs; no model libraries |
| `requirements/dataset-clip.txt` | Optional OpenCLIP ordering of new positive tasks |
| `requirements/evaluation.txt` | Evaluate saved `.pt` rankings without encoder libraries |
| `requirements/dev.txt` | CPU tests and linting |

Set `data.dino_cache`, `data.cache` (FAFA), `data.clip_cache` and image paths in
the proposed YAML. Completed compatible DINO/FAFA caches are reused; the CLIP
stage builds only missing frozen semantics.
`build-cache --cache-stage dino|persons|clip` runs a single stage; default `all`
runs all three. Interrupted builds resume completed image shards.

```bash
# Build/reuse caches, warm up grounding/identity, train, validate.
bash scripts/methods/proposed.bash val

# Fixed checkpoint, coarse / top-500 fine / full-gallery fine controls.
python tools/run.py ablate --config configs/ablations/shortlist.yaml --splits val

# CLIP and official FAFA baselines keep their workflows.
bash scripts/methods/clip.bash val
bash scripts/methods/fafa.bash val
python tools/report.py --split val --run proposed=runs/proposed-v2/val
```

For a frozen test run, use `bash scripts/methods/<method>.bash test`. Proposed
loads `runs/proposed-v2/run_config.yaml`, saved by the validation workflow, and
its selected `best.pt`; CLIP/FAFA reuse their validation selections. Test
workflows never train or select weights. For a custom config, stage or override,
call `tools/run.py` directly. See [workflow contracts](scripts/README.md).

The new architecture requires training from scratch. Source IDs/checksums,
per-image detector boxes, gallery order, encoder/preprocessing metadata and
checkpoint/cache versions are checked. Legacy DINO metadata is rejected by
default; explicitly assert known original settings with
`cache.allow_legacy_dino=true` only when appropriate.

| Command | Purpose |
| --- | --- |
| `prepare` | Prepare official FAFA or baseline assets |
| `build-cache` | Build/reuse DINO, FAFA and CLIP stages |
| `train` | Integrated warmup/main training; overall validation Full-mAP selects best.pt |
| `retrieve` | Save complete split-gallery rankings and runtime metadata |
| `evaluate` | Evaluate saved rankings without loading a model |
| `run` | Coordinate prepare/build/train/retrieve/evaluate |
| `ablate` | Compare explicit v2 retrieval policies |

`--set key=value ...` overrides existing fields. `--splits` selects query splits.
`--max-queries` is for matching retrieve/evaluate smoke subsets. Training writes
`warmup.pt`, `last.pt`, selected `best.pt`, config, history and per-epoch metrics;
retrieval writes rankings and provenance. Train into a new output directory;
optimizer resume is not implemented. W&B is optional and disabled by default.

See [Kaggle commands and limitations](docs/proposed_runs.md),
[baseline workflow](docs/baselines.md), and [retrieval controls](docs/ablations.md).

## Tests

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q
python -m ruff check src tools tests labelstudio
```

The CPU contract suite also runs in GitHub Actions on Python 3.11 and 3.12.
`dev` includes SciPy for the FAFA matching adapter tests; native FAFA weights
are not needed.

Tests cover categorical grounding, long text/all mentions, transport constraints
and an independent optimum, null handling, gradient ownership, permutation
invariance, AMP edge cases, cache resume/reuse and the full train/retrieve/evaluate
pipeline. A tiny synthetic overfit checks learning, not real retrieval quality.
Optional real CLIP/DINO implementation checks use small random models. Real T4
FP16, pretrained-backbone integration, RCR mAP and speed/memory remain to be measured.

For baseline notebook setup, see [baseline setup](docs/baselines.md).
The deterministic dataset pipeline uses `requirements/dataset.txt`; optional
candidate ordering uses `requirements/dataset-clip.txt`. Report generation reads
JSON and works with the base package; decoding saved rankings needs the
evaluation profile. No model client is required for annotation preparation.

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
