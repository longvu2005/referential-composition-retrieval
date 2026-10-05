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

## Setup

Use Python 3.11 or 3.12 and run commands from the repository root. Keep proposed,
CLIP and FAFA in separate environments because FAFA pins an older Transformers.
Example for proposed:

```bash
python3.11 -m venv .venv-proposed
source .venv-proposed/bin/activate
python -m pip install -r requirements/bootstrap.txt
python -m pip install --no-build-isolation -r requirements/proposed.txt
```

For CLIP/FAFA replace `proposed` with `clip`/`fafa` in the environment and
requirements filename. On Kaggle, call that environment's Python explicitly
instead of relying on activation carrying over between cells.
`requirements.txt` still installs the combined proposed + dataset dependencies;
`requirements/proposed.txt` now installs only proposed dependencies.

## Run experiments

Edit data/checkpoint/cache paths in the method YAML first. All relative paths
are relative to the working directory, which should be the repository root.
The scripts never install packages automatically.

```bash
# Proposed: build cache, train, select best.pt on val, evaluate val and test.
python tools/methods/run.py run --config configs/methods/proposed.yaml --build-cache --train

# Reuse the cache and train a new model.
python tools/methods/run.py run --config configs/methods/proposed.yaml --train

# Evaluate an existing checkpoint.
python tools/methods/run.py run --config configs/methods/proposed.yaml

# CLIP: prepare one image/text checkpoint, run all four modes.
python tools/methods/run.py run --config configs/methods/clip.yaml --prepare

# FAFA: use its own environment.
python tools/methods/run.py run --config configs/methods/fafa.yaml --prepare
```

| Command | Purpose |
| --- | --- |
| `prepare` | Download/verify CLIP or official FAFA assets |
| `build-cache` | Build proposed's frozen visual cache |
| `train` | Train proposed from its cache |
| `retrieve` | Save rankings for the selected split(s) |
| `evaluate` | Evaluate saved rankings without loading a model |
| `run` | Retrieve + evaluate val/test; optionally prepare/build/train first |
| `ablate` | Run a proposed ablation suite; optional beta sweep selected on val |

`--splits val` or `--splits test` restricts the run. `--modes clip_image clip_text`
selects CLIP modes. `--set key=value ...` overrides existing YAML fields without
modifying the file:

```bash
python tools/methods/run.py run --config configs/methods/proposed.yaml --train \
  --set data.image_root=/kaggle/input/pipa/images data.cache=/kaggle/input/rcr-cache/cache
```

CLIP `run` tunes early/late fusion separately on complete **val**, records the
selected image/text weights, then applies them unchanged to test. A test-only
fusion run requires the matching saved val selection. Low-level `retrieve`
uses the fixed weights in YAML; use `run` for automatic val selection.

`output.dir` is a root directory, with no `{split}` / `{mode}` templates:

- Proposed: `runs/proposed/{val,test}/` plus checkpoints, tokenizer and training logs.
- CLIP: `runs/clip/<mode>/{val,test}/`, `tuning.json` and `summary.csv`.
- FAFA: `runs/fafa/{val,test}/` and `summary.csv`.

Each split saves `rankings.pt`, `run.json`, `metrics.json`. Baselines also save
`scores.npy`. `summary.csv` describes the latest successful `run` invocation.

Detailed usage: [proposed](docs/proposed_runs.md), [baselines](docs/baselines.md).
Coarse normalization, beta sweep and extensible ablations:
[Vietnamese guide](docs/ablations_vi.md).
Identity amplitude A–D, fixed shortlist, and validation/seed confirmation:
[identity balance guide](docs/identity_balance_ablation.md).
Model equations: [proposed method](docs/proposed_method.md).
Training fixes and data review: [research notes](docs/research_fix_vi.md).

Training uses same-identity negatives mixed with random negatives. Optional
coarse mining runs only on train after warmup. The state loss compares images
containing all required identities; wrong-identity images are ignored by that
loss. Conflicting negatives from equivalent train instructions are excluded
without editing the reviewed labels or the evaluation protocol.

Read-only label and optional cache coverage audit:

```bash
python tools/dataset/audit.py --output runs/data_audit.json
python tools/dataset/audit.py --cache cache/proposed --output runs/cache_audit.json
```

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tools tests labelstudio
```

Core tests use small local encoders and cache fixtures. CLIP's native checkpoint
integration tests need the `clip` extra; baseline preparation tests need `gdown`.
The CUDA memory test is skipped on CPU. Tests do not download full pretrained
models or measure PIPA retrieval quality.

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
bash scripts/phase2_finalize.bash --version 0.1.0
```

`phase2_finalize.bash` validates positive decisions and writes the deterministic
final export. It does not call Gemini for already accepted Stage-2 samples.
