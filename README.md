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

## Repository layout

```text
referential-composition-retrieval/
├── configs/methods/proposed/    proposed-method experiment configs
├── dataset/                     dataset source, work files, final export
├── labelstudio/                 local review / positive-set UIs
├── scripts/                     dataset shell launchers
├── src/rcr/
│   ├── dataset/                 dataset schemas and construction
│   ├── evaluation/              official ranking protocol and metrics
│   ├── methods/common/          shared data, detection, anchor matching
│   └── methods/proposed/        proposed RCR model
├── tools/dataset/               dataset CLIs
├── tools/methods/               cache/train/retrieve/evaluate CLIs
└── tests/                       dataset, method, and evaluation tests
```

The supported dependency direction is:

```text
scripts/ -> tools/ -> src/rcr/
```

`src/rcr/` never imports from `tools/` or `scripts/`.

### Reading and editing the research code

Start with `tools/methods/train_proposed.py`: setup, batches, loss/backward,
periodic retrieval evaluation, then checkpoint saving. The loop uses ordinary
functions and PyTorch modules, with no trainer or callback framework.

| What to inspect or change | Main location |
| --- | --- |
| Text tokenization and all Subject mentions | `encoders.py::encode_query_text` |
| Cached features, identity labels, positive masks | `batch.py::build_batch` |
| Query grounding and composition | `model.py::RCRModel.encode_query` |
| Target binding and fine score | `model.py::RCRModel.score_target` |
| Four losses and their supervision masks | `training.py::compute_loss`, `losses.py` |
| Coarse shortlist and fine reranking | `retrieval.py::retrieve_rankings` |
| Optimizer, W&B, evaluation schedule, checkpoints | `tools/methods/train_proposed.py` |

Module filenames in this table are relative to `src/rcr/methods/proposed/`.
Training and retrieval share text preparation and query encoding; edit these
once to keep both paths aligned. `RCRModel.forward` scores one query-target pair
per batch row, while training and retrieval reuse one encoded query for several
targets. Each mathematical component remains a small module so its tensors and
equations can be inspected directly. Existing model dimensions, cache files, metric
names, and model checkpoint parameter names are preserved by this refactor.

---

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

---

## Proposed method

The implemented inference path is:

```text
person detector -> shared image encoder -> Subject grounding
-> coarse identity + state shortlist -> structured identity-set composition
-> target evidence binding -> fine target-set reasoning -> ranking
```

### 1. Shared visual representation

A `person`-prompted detector supplies every detected person candidate; detected
heads do not gate candidates. A shared image encoder `E_I` produces:

- letterboxed whole-scene patch features `F`;
- full-person crop features `h_i`.

A trainable identity head produces normalized identity embeddings:

```text
v_i = Normalize(P_id(h_i))
```

The expensive image encoder is frozen behind the feature cache. `P_id` remains
inside `RCRModel` and is trained normally; identity embeddings are therefore
**not** stored permanently in the cache. Scene-box coordinates are transformed
to the same letterboxed coordinate system as the scene patch grid. GT head
annotations are used only to align optional identity supervision and compute
evaluation labels; no head crop enters the model.

### 2. Shared evidence binding

One `EvidenceBinding` module is shared by query grounding and target evidence
binding. Its order is **Scene → Reference → Person → Conditioned Scene**.

First, reference tokens condition the scene patches:

```text
F_bar = LN(F + Attn(F, R, R))
```

Then each person reads that conditioned scene using a soft geometry bias from
its box in letterboxed scene coordinates:

```text
e_i = Attn_geo(h_i, F_bar, F_bar; B_i)
b_i = FFN_res(h_i + e_i)
```

The geometry is a soft attention bias, not a hard crop mask, so relational and
contextual evidence may remain outside the person box.

### 3. Query Subject grounding

For Subject `s`, the shared text encoder `E_R` encodes its selection text. The
same encoder processes the change text; the shared Binding produces one raw
grounding logit per query person:

```text
g_si = Ground(F_q, h_i^q, B_i^q, T_sel^s)
m_si = sigmoid(g_si)
```

Membership uses independent sigmoid probabilities, never a softmax across
persons; a GROUP Subject can therefore select multiple people. Training uses
masked `BCEWithLogits` on raw finite logits.

### 4. Cheap coarse identity + global state retrieval

Coarse retrieval is deliberately optimistic and is used only to preserve high
candidate recall.

```text
c_i(q,t) = max_j dot(v_i^q, v_j^t)
a_si = sigmoid(g_si)
omega_si = a_si / (sum_r a_sr + eps)
S_subject_s(q,t) = sum_i omega_si c_i(q,t)
S_id(q,t) = average_s S_subject_s(q,t)
z_text(q) = Normalize(P_text(masked_mean(E_R(final_change))))
z_image(t) = Normalize(P_image(mean_patches(F_t)))
S_state(q,t) = dot(z_text(q), z_image(t))
S_coarse(q,t) = S_id(q,t) + beta * S_state(q,t)
```

Each Subject contributes once to the average, including when one Subject
contains multiple people. Invalid person/Subject positions are masked. There
is no threshold, one-to-one assignment, or coverage heuristic in `S_id`; its
soft grounding + max identity similarity formula is unchanged. The state branch
uses only the existing `final_change` token encoding (including Subject markers),
with padding excluded from mean pooling, and whole-image patch means. Separate
trainable text/image projections map both to `model.state_dim`, followed by L2
normalization. State is a rough global action/context signal; precise identity
binding is handled by the fine reasoner. `model.coarse_beta: 0.3` is a starting
value to tune on validation, not a measured optimum. Top `M` gallery images
proceed to the fine stage. Set beta to zero for identity-only shortlisting.

### 5. Structured identity composition

The change text contains distinct `[S1]` / `[S2]` tokenizer special tokens
aligned with Subject IDs. Each separate identity token uses the reference-free
query identity `v_i` plus the embedding for its Subject role.

Independent grounding probabilities are normalized only when a Subject marker
needs a Subject-level identity summary; the individual identity tokens remain
available to the fine reasoner. Identity-token keys receive the additive
`log(sigmoid(g_si))` membership prior in composition attention, target scene
binding, and fine self-attention. CLS/change keys have neutral prior; padding
is masked. No hard membership threshold removes a valid identity token.

This gives:

- permutation invariance among identities inside one Subject;
- explicit role distinction between Subjects; and
- individual identity tokens for final target matching.

### 6. Target evidence binding and fine reasoning

For each coarse Top-M target, the same `EvidenceBinding` conditions its scene
with the **composed reference** (change, Subject roles, individual identity
tokens, and aligned membership prior). Target person tokens combine:

```text
LN(projected bound evidence + projected target person identity
   + projected scene-box geometry)
```

The composed query cross-attends to the whole target person set, then a
self-attention block scores its CLS token as the raw scalar `S_f(q,t)`.

The final Top-M order is determined only by `S_f`; the coarse score is used only
for shortlisting. Images outside Top-M retain their coarse order so the saved
output remains a complete split-gallery ranking. No coarse, identity, or coverage
score is manually added to the fine score.

When the query has no detected people, it remains in retrieval and evaluation:
`S_id` is zero for nonempty target person sets, so global state can still
rank them. Empty target sets retain the original `-inf` identity/coarse score. The fine stage can still use change text
and target evidence. Empty targets and padded persons/Subjects do not crash.

### Training objective

The current objective is:

```text
L = 1.0 * L_ret + 1.0 * L_ground + 0.1 * L_id + 1.0 * L_state
```

where:

- `L_ground`: masked BCE-with-logits for independent Subject/person labels;
  a Subject whose annotated identities have no detected match is excluded from
  grounding supervision instead of treating every detection as a negative;
- `L_id`: supervised contrastive loss (default temperature `0.1`) over query
  persons and persons from valid positive target images, using one shared identity
  label vocabulary. Unknown identity `-1`, padding, negative target images and
  self-pairs are excluded. Repeated copies of the same cached image/person crop
  count once per batch; different images of the same identity remain positives;
- `L_ret`: mean pairwise `softplus(S_f(q,n) - S_f(q,p))` over valid
  positive/negative pairs;
- `L_state`: the same masked pairwise ranking loss applied to
  `S_state / state_temperature` (default `0.1`), using the sampled Full Positive
  and negatives. It trains both state projections and the shared text encoder;
  there is no in-batch negative assumption. `state_weight` defaults to `1.0`.
  Fine ranking uses only `S_f`; neither state nor identity is added to its score.

Candidate sampling picks one reviewed positive and excludes **all** other
known positives and the query image from that query's negatives.
Including positive targets supplies cross-image identity pairs even when query
identities do not repeat within a batch. `L_id` can still be zero when detected
persons lack known matching identities; target images are not assigned the query's
labels by assumption. GT-aligned cache labels determine identity matches.

---

## Setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
```

Verify the package:

```bash
python -c "import rcr; print(rcr.__file__)"
```

The default image config uses Meta's DINOv3 ViT-B/16 checkpoint on Hugging
Face. That checkpoint is gated, so accept its license and authenticate with
Hugging Face before building the cache.

Run the test suite and check the proposed implementation's source style:

```bash
pytest -q
ruff check src/rcr/methods/proposed \
  tools/methods/build_cache.py \
  tools/methods/train_proposed.py \
  tools/methods/retrieve_proposed.py
ruff format --check src/rcr/methods/proposed \
  tools/methods/build_cache.py \
  tools/methods/train_proposed.py \
  tools/methods/retrieve_proposed.py
```

---

## Proposed-method configs

All experiment parameters live under:

```text
configs/methods/proposed/
├── build_cache.yaml
├── train.yaml
├── retrieve.yaml
└── evaluate.yaml
```

Do not hard-code experiment hyperparameters in the CLI scripts.

### `build_cache.yaml`

Controls:

- final dataset and raw-image paths;
- cache output path;
- Grounding DINO model and thresholds;
- shared DINOv3 model;
- scene/person letterbox sizes;
- `cache.storage_dtype`: `float32` (default) or `float16` for saved scene/person
  features and the gallery person index;
- device.

The default `scene_size: [224, 224]` produces a 14x14 patch grid for a ViT/16
backbone. Changing it changes the experiment and requires rebuilding the cache.

For limited disk space, set:

```yaml
cache:
  storage_dtype: float16
```

This halves feature payload storage relative to compact FP32, with FP16 rounding.
Loaded training/fine-scoring features are cast to FP32; coarse gallery features
are cast to the identity head's dtype before projection. Boxes stay FP32 and
identity labels are unchanged. For 37,107 images with 196 patches and 768
dimensions, scene features alone need about **20.8 GiB in FP32 / 10.4 GiB in
FP16**, plus persons, the index and file overhead. This option does not change
the patch grid, but retrieval quality should be checked when changing precision.

Global image features use mean pooling of the **stored** whole-image patch
features, cast to FP32 before averaging. New caches include `global_features`
`[G,D]` in `index.pt`. Existing caches remain usable without rerunning detection
or image encoding: the first state retrieval reads each feature file once,
keeps only its mean vector in CPU RAM, and reuses those vectors for subsequent
evaluations in the same process. This initial disk scan can take time on large
caches; it does not write to read-only Kaggle input directories. Only projected
identity/state chunks go to the accelerator. Projected state vectors are
recomputed after model updates, while frozen raw global vectors are reused.

### `train.yaml`

Controls model dimensions, optimizer parameters, loss weights, identity
temperature, training candidate count, seed, device, and
output directory, plus `model.state_dim`, `model.coarse_beta`,
`loss.state_weight` and `loss.state_temperature`. The state ranking loss trains
the coarse state branch; the fine loss remains unchanged. The
`evaluation` block controls periodic retrieval evaluation: interval, fixed train
subset size, coarse shortlist, scoring batch sizes and CandidateRecall cutoffs.
The complete validation split is evaluated at the interval and at the final
epoch. `best.pt` is selected by validation `Full-mAP`.

The optional `wandb` block controls experiment tracking: project/run name,
online/offline mode, and step logging interval. Tracking records only the
model/train/loss/optimizer/evaluation configuration, cache identity/shape,
losses, learning rates, gradient norm, identity-loss activity, grounding
supervision rate, and aggregate train/validation retrieval metrics. It does not
upload images, cache features, model graphs, rankings, or checkpoints.

The current training sampler uses **one reviewed Full Positive plus random
negatives** per query. Batches are grouped by number of Subjects, so every batch
has a fixed Subject axis. Candidate count is fixed by `train.candidates`.
Both positives and negatives come only from the complete **train image split**.
Train diagnostics search the train gallery; validation searches the val gallery.
There is no configurable cross-split negative pool.

### `retrieve.yaml`

Controls checkpoint, split, coarse `top_m`, fine-scoring batch size, identity
projection/coarse batch sizes, device, and ranking output directory. The state
dimension and beta come from the checkpoint training config, ensuring periodic
validation and standalone retrieval use the same formula and split protocol.
The reference-free cache remains shared and can be reused without rebuilding.
Saved ranking indices are local to their saved `gallery_ids`, so rankings made
with the old global gallery must be regenerated. Old checkpoints lack
the state projections and require retraining; they are rejected explicitly.

### `evaluate.yaml`

Controls evaluation split, saved ranking path, CandidateRecall cutoffs, and
metrics output path.

---

## End-to-end proposed-method run

Run all commands from the repository root.

### 1. Build the reference-free visual cache

```bash
python tools/methods/build_cache.py \
  --config configs/methods/proposed/build_cache.yaml
```

The cache stores reference-free scene patches, person features and scene-aligned
boxes, plus optional GT-aligned identity labels for supervision. It stores no
head crop features or fixed `P_id` embeddings. Retrieval does not consume GT
identity labels as model input.

Each saved tensor owns compact CPU storage. In particular, the person CLS view
is copied once and that compact tensor is reused for the feature file and the
gallery-index accumulator, so patch tokens are not retained through the CLS view.
Scene patches are also copied without unused CLS/register storage.

Cache layout:

```text
cache/proposed/
├── index.pt
└── features/
    ├── 0.pt
    ├── 1.pt
    └── ...
```

The cache builder marks an in-progress build and records one build ID in the
index and each feature file. Training and retrieval reject incomplete builds,
mixed feature files, or a gallery that differs from the finalized dataset.
New checkpoints require the same cache build at retrieval. Legacy caches and
checkpoints without build IDs remain readable but cannot be cross-checked this
way. After an interrupted build, rerun cache building to completion before use.

Rebuild the cache whenever the detector, image backbone, or preprocessing
changes. **Old head-based caches and checkpoints are incompatible with this
implementation**: rebuild the cache and retrain before retrieval. Retraining
`P_id` after this migration does not require a cache rebuild because the
current projection is applied to cached person features at runtime.

Existing person-based FP32 caches remain readable, but their files do not shrink
automatically. Rebuild them to obtain compact storage or change storage precision;
building to a separate directory temporarily requires space for both caches.
The query-plus-positive-target identity loss can use an existing cache with
GT-aligned `identity_ids` without rebuilding. It takes effect on subsequent
training steps, not on weights already learned by an old checkpoint.

### 2. Train

When W&B is enabled, authenticate once before training. On Kaggle, store
`WANDB_API_KEY` as a secret; use `wandb.mode: offline` when the notebook has no
network access. Set `wandb.enabled: false` to disable tracking entirely.

```bash
python tools/methods/train_proposed.py \
  --config configs/methods/proposed/train.yaml
```

Outputs are written under the configured run directory, by default:

```text
runs/proposed/
├── config.yaml
├── tokenizer/
├── evaluation/
│   └── epoch_NNN/
│       ├── train_metrics.json
│       └── val_metrics.json
├── best.pt
└── last.pt
```

`config.yaml` is copied into the run directory and is also embedded in the
checkpoint for reproducibility. `last.pt` is updated after every epoch;
`best.pt` is updated only when validation `Full-mAP` improves. Periodic train
metrics use one deterministic subset chosen from `evaluation.train_max_queries`.
Only aggregate metrics are saved during training, so full rankings do not consume
additional run storage. If periodic evaluation is disabled, use `last.pt` in the
retrieval config because no `best.pt` is produced.

### 3. Retrieve / rerank

```bash
python tools/methods/retrieve_proposed.py \
  --config configs/methods/proposed/retrieve.yaml
```

Default output:

```text
runs/proposed/test/rankings.pt
```

The file stores:

```text
sample_ids
gallery_ids
rankings       # full split-gallery ranking, int32 indices
coarse_topm    # coarse shortlist, int32 indices
```

`gallery_ids` lists the complete requested split in canonical registry order.
`rankings` contains every image in that gallery except the query exactly once.
Evaluation rejects saved rankings whose gallery differs from the requested split.

### 4. Evaluate

```bash
python tools/methods/evaluate_proposed.py \
  --config configs/methods/proposed/evaluate.yaml
```

Default output:

```text
runs/proposed/test/metrics.json
```

The evaluator reports:

- `ID-mAP`, `ID-R@1/5/10`;
- `Full-mAP`, `Full-R@1/5/10`;
- `CandidateRecall@K` for the configured coarse-shortlist cutoffs;
- aggregate metrics by case type;
- per-query metrics.

`Full-mAP` is the primary full-task metric.

---

## Cache and supervision rules

The method follows these separation rules:

- raw benchmark data is immutable;
- detector/image-encoder outputs are reference-free cache artifacts;
- query/target identity embeddings are produced by the current trainable
  `P_id`, not permanently cached;
- GT head identity IDs may supervise training but are not consumed by the
  retrieval path;
- the query image is excluded from its own ranking;
- Full Positive definitions and evaluation code are method-independent;
- experiment configs live under `configs/methods/proposed/`; a config copy and
  model checkpoint are saved under the configured `runs/` directory.

The detector retains all detected person candidates, including people with no
detected head. GT heads can align supervision labels to predicted persons;
unmatched persons have unknown identity labels. Person-detector recall still
limits what grounding can select and should be measured before a large run.

---

## Evaluation protocol

The same saved final ranking is scored against two positive sets.

### Identity retrieval

- `ID-mAP`
- `ID-R@1`
- `ID-R@5`
- `ID-R@10`

An ID Positive contains every identity required by all Subjects, regardless of
the requested target change.

### Full RCR retrieval

- `Full-mAP`
- `Full-R@1`
- `Full-R@5`
- `Full-R@10`

A Full Positive is one of the reviewed `positive_image_ids` for the query.

### Coarse diagnostic

`CandidateRecall@K` measures the fraction of reviewed Full Positives retained
inside the coarse top-K shortlist. It is a proposed-method diagnostic and does
not replace the final retrieval metrics.

---

## Current implementation status

The repository contains tested implementations for:

- person detection with optional GT-head-to-person label alignment;
- reference-free feature caching;
- query Subject grounding;
- soft coarse identity + global state retrieval;
- structured identity-set composition;
- target evidence binding and fine reasoning;
- training candidate sampling and losses;
- config-driven training, retrieval, and evaluation CLIs.

The source-level test suite currently covers these contracts. A successful unit
test run does not replace a real pretrained-model smoke run on the target
machine; cache size, detector coverage, GPU memory, and throughput must still
be verified before launching the full experiment.
