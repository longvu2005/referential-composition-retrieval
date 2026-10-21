# Referential Composition Retrieval

Minimal research code for **Referential Composition Retrieval (RCR)**: given a
query image, text that identifies one or more Subjects, and a requested change,
rank gallery images that preserve the referenced identities and satisfy the
change.

The benchmark contains four query types:

- `INDIVIDUAL`: one Subject, one identity;
- `GROUP`: one Subject, multiple identities;
- `DUAL`: two Subjects with separate changes;
- `RELATIONAL`: two Subjects linked by an ordered relation.

Inference uses the reference image, `final_desc`, `final_change` and public
`subjects[].subject_id` slots; CLIP uses `final_instruction` by default. Subject
slots are query-local references, not identity labels or group sizes. Retrieval
scoring does not use `case_type`: FAFA predicts members with the same CLIP rule
for every query and splits conditions only when the text has independent clauses.

GT head boxes, identity labels, seed targets and positives provide train
supervision or evaluation labels. Validation selects settings; test labels are
used only for evaluation and provenance, never retrieval scoring.

| Boundary | Data available |
| --- | --- |
| Feature extraction / ranking | Images, text, public Subject slots, predicted boxes |
| Proposed training supervision | Train GT alignment, identities and Full Positive labels, separately from features |
| Evaluation / provenance | Annotated samples, GT heads/identities, case labels and positive sets |

`load_rcr_inputs()` drops annotation fields and never opens `head_boxes.jsonl`.
Runners keep annotated data for evaluation/provenance and pass public views to
scorers. `case_type` remains useful for train sampling and by-case reporting.

## Method

The proposed model uses frozen DINO scene features, frozen native FAFA person and
query features, learned Subject grounding, one global partial identity matching,
and a Context Reasoner. Matching includes a null option, so missing people are not
forced onto unrelated detections. The same assignment is reused by FDA binding and
context reasoning.

Coarse retrieval combines normalized identity and global-state scores. Fine
retrieval reranks the coarse Top-M shortlist with native FAFA binding and learned
relational context. Validation selects the checkpoint and the two retrieval fusion
weights; test labels are never used for model or weight selection.

See [method equations](docs/proposed_method.md),
[benchmark protocol](docs/benchmark_protocol.md), and
[experiment details](docs/proposed_runs.md).

## Layout

```text
configs/                 experiment configurations
src/rcr/common/          shared data, I/O and runtime helpers
src/rcr/dataset/         deterministic dataset construction
src/rcr/evaluation/      metrics and report generation
src/rcr/baselines/       CLIP and native FAFA baselines
src/rcr/proposed/        cache, training and retrieval code
tools/run.py             single experiment CLI
scripts/                 reproducible setup and run recipes
tests/                   CPU contract tests
```

Dependency direction is CLI → experiment functions → model/shared utilities.
There is no trainer framework, plugin registry, or recursive configuration system.

## Setup

Use Python 3.11 or 3.12. Each method has its own environment because native FAFA
and the proposed model require incompatible Transformers versions.

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash fafa
PYTHON=python3.11 bash scripts/setup.bash clip   # only for CLIP baselines
```

Available environments are `proposed`, `clip`, `fafa`, `dataset`,
`dataset-clip`, `evaluation`, and `dev`. Dependency versions live in
`pyproject.toml`; files under `requirements/` only select an extra.

## Run

The main validation workflow builds or reuses caches, trains, evaluates, and
calibrates the two retrieval weights:

```bash
bash scripts/methods/proposed.bash val
```

Run the frozen test configuration only after validation:

```bash
bash scripts/methods/proposed.bash test
```

Baselines follow the same validation-first protocol:

```bash
bash scripts/methods/clip.bash val
bash scripts/methods/fafa.bash val
```

Lower-level stages use one CLI:

```bash
python tools/run.py prepare     --config configs/proposed.yaml
python tools/run.py build-cache --config configs/proposed.yaml
python tools/run.py train       --config configs/proposed.yaml
python tools/run.py retrieve    --config configs/proposed.yaml --splits val
python tools/run.py evaluate    --config configs/proposed.yaml --splits val
python tools/run.py calibrate   --config configs/calibration.yaml
```

Use `--set key=value ...` for typed overrides. `--max-queries` is only a smoke-test
option for matching retrieve/evaluate runs; normal results use complete splits.

Training prints the actual device/AMP setting and average input-wait, forward and
backward times per step. `runtime.json` records the selected hardware; history
separates training, mining, evaluation and checkpoint time. On Kaggle, set
`runtime.device=cuda` to require a GPU. See the
[cache I/O audit](docs/cache_io_audit.md).

Large cache indices stay memory-mapped. Scene/target shards use buffered reads
by default (`cache.mmap_shards=false`), matching the original reader. Legacy
global means are loaded only for the gallery being ranked. To compare read
modes on your own cache without training:

```bash
.venv-proposed/bin/python tools/profile_cache.py --config configs/proposed.yaml --batches 1
```

### Caches

The proposed pipeline keeps three independent visual caches:

- `data.dino_cache`: DINO scenes and detections;
- `data.fafa_image_cache`: FAFA person features and target tokens;
- `data.fafa_query_cache`: multimodal query features.

DINO detections fix the person boxes and crop order. FAFA image and query
extraction use those same boxes on the original RGB images; GT never changes a
crop or selects which people enter a visual cache. Training reads the cached
features and does not open or crop the original images.

Completed caches are reused automatically. Build only one stage with
`--cache-stage dino`, `fafa-image`, or `fafa-query`. Query text changes invalidate
only the query cache. FAFA's text budget comes from `checkpoint.max_txt_len` in
`configs/fafa.yaml`. Use a separate query-cache directory
when changing that policy; image features remain reusable.

`cache.lru_mib` bounds scene reads; `cache.native_lru_mib` independently bounds
FAFA target/query shards. `cache.text_lru_mib` bounds frozen BERT output tensors;
its projection is recomputed and trained every step. A zero budget disables that
memory cache. Cached tensors retain their original precision, and FDA normalizes
matched target rows in FP32.

Visual caches do not store GT labels. Train-only crop-to-ID alignment is prepared
before epoch 1 and reused from `cache.supervision_cache`. With
`train.identity_sampling.max_crops > 0`, this covers the complete train gallery.
Compatible query/positive labels from older runs are retained; only missing
train images are aligned. With auxiliary sampling disabled, only query and
Full Positive images need preparation.
Prepare the labels separately before launching a training job with:

```bash
.venv-proposed/bin/python tools/run.py prepare-supervision --config configs/proposed.yaml
```

This command needs completed DINO/FAFA image caches but no encoder weights or
FAFA query cache. It reads crop boxes through mapped shards without copying
full scene/person feature payloads. `train` also prepares any missing labels
automatically, before loading BERT or starting an epoch. Training batches only
look up these prepared labels; they do not align heads, open raw images, or
rewrite the supervision file.
Changed cache IDs, train heads or scene size invalidate the supervision cache.
The alignment algorithm, visual crops and retrieval candidate sampling remain
unchanged.

Identity loss keeps the unique annotated query/Full Positive crops and adds a
bounded auxiliary batch from the train gallery. Each query Subject ID can draw
up to `crops_per_identity` extra observations from different images, regardless
of retrieval conditions or whether other Subjects co-occur. Up to
`random_identities` additional train IDs supply more identity classes, including
IDs absent from the query set. Random IDs must have observations in at least two
images. Already-used query/positive images and duplicate crops are excluded.
Unknown/unmatched detector crops never enter this pool.

The default config requests two crops per ID and eight additional random IDs,
with at most 128 extra crops per step. Short pools use fewer crops. Features come
only from the pooled FAFA person index and run through the shared person/identity
projection; auxiliary images do not load scenes or native query/target tokens,
or enter fine retrieval. Identity sampling has its own seeded generator so
retrieval candidate draws are unchanged. `max_crops: 0` restores the original
query/positive-only identity objective; older configs without this section also
retain that behavior. `identity_weight: 0` skips the auxiliary pool entirely.

Training logs `identity_extra_crops`, `identity_classes_per_batch`,
`identity_valid_anchor_rate` (observations with another same-ID crop), and
`identity_positive_pairs` (unordered pairs). Use validation Full-mAP and matching
quality to compare the new objective with `max_crops: 0`; more identity data does
not guarantee better retrieval.

Older cache files remain readable; their `identity_ids` are discarded by the
feature API. To remove that metadata physically, run this once on a **writable
DINO cache**:

```bash
.venv-proposed/bin/python tools/data/clean_cache.py --cache /path/to/dino-cache
```

This preserves tensors, crop order and cache ID, so FAFA image/query features do
not need extraction again. Legacy caches without pixel boxes additionally need
raw image sizes and the original scene size when creating train supervision.

## Dataset

The checked-in snapshot is version `0.1.0` with 4,311 queries and 37,107 images.
Each query searches all images in its own PIPA split, excluding its query image.

| Split | INDIVIDUAL | GROUP | DUAL | RELATIONAL | Total |
|---|---:|---:|---:|---:|---:|
| train | 3,472 | 137 | 269 | 155 | 4,033 |
| val | 194 | 12 | 36 | 22 | 264 |
| test | 5 | 2 | 3 | 4 | 14 |
| **all** | **3,671** | **151** | **308** | **181** | **4,311** |

The current test split is a pipeline-validation split, not a publication-scale
holdout. Raw images are not stored in the repository; configure `data.image_root`
before extraction. Dataset reconstruction is documented in
[dataset/README.md](dataset/README.md).

Audit Full Positive integrity, equivalent-query conflicts and review scope:

```bash
.venv-dev/bin/python tools/data/audit.py \
  --output runs/data_audit.json \
  --review-queue runs/full_positive_review.jsonl
```

Add `--positive-catalog` and `--positive-decisions` with the original work files to
check their candidate scope against every split ID Positive. Missing review
artifacts mean **unknown scope**, not proven complete labels. The audit writes
reports and a separate review queue; it never changes annotations or trains a
model. See [Full Positive audit](docs/full_positive_audit.md) for findings,
manual-review commands and the limits of this check.

## Tests

```bash
bash scripts/setup.bash dev
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/python -m ruff check src tools tests labelstudio
```

Tests use tiny local backbones and caches. Native FAFA weights and pretrained model
downloads are not required for the CPU suite; CUDA-only numerical and memory checks
skip automatically when no GPU is available.
