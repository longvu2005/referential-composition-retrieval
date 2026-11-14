# Proposed method: experiment protocol

Run every command from the repository root. Use Python 3.11 or 3.12 and keep the
proposed (PyTorch >=2.6) and FAFA (PyTorch 2.5.1) environments separate.

## 1. Configure paths

Edit `configs/proposed.yaml`:

```yaml
data:
  final_dir: dataset/data/final
  image_root: /path/to/images
  detection_cache: /path/to/cache/detections
  scene_cache: /path/to/cache/scene
  fafa_image_cache: /path/to/cache/person-images
  fafa_query_cache: /path/to/cache/person-queries
person_encoder:
  python: .venv-fafa/bin/python
cache:
  supervision_cache: /path/to/cache/train-supervision.json
```

All model, optimizer, sampling, retrieval and runtime choices live in this one
file. Command-line overrides must name an existing key.

## 2. Prepare frozen assets and caches

```bash
python tools/run.py prepare --config configs/proposed.yaml
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage detections
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage scene
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-image
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query
```

`build-cache` without `--cache-stage` runs all missing stages. Detection owns pixel
boxes, original image sizes and crop order. CLIP owns scene/global features only.
Each of these two stages can run independently. The FAFA image cache owns query-independent person
means and target tokens. The FAFA query cache owns text-conditioned query
features. Editing query text therefore invalidates only the last stage; changing
the CLIP scene encoder does not invalidate detection or either FAFA cache.

The FAFA worker runs through `person_encoder.python`, so native FAFA never enters
the proposed environment. Training and retrieval read cache files and do not load
FAFA weights.

FAFA query extraction keeps the complete condition for each Subject and anchors
its mentions to “the reference person”, without splitting clauses. Content-based
keys reuse unchanged condition shards and reject stale query entries. Rerun the
query stage after editing query text; detection, scene and FAFA image/target
caches remain reusable. Checkpoint scoring compatibility is described in the
[feature contracts](feature_contracts.md#checkpoint-compatibility).

## 3. Train and select a checkpoint

```bash
python tools/run.py train --config configs/proposed.yaml
```

Training has separate grounding, crop-identity, coarse-identity, conditional
context and global-state objectives. Context ranks the shared score
`alpha * B_FAFA + beta * (coverage * tanh(context) - (1 - coverage))`, using
`beta = retrieval.context_weight` and `loss.context_temperature`. FDA binding and
the binding scale are frozen. Pairwise
losses weight eligible queries equally; see the exact objectives in
[proposed_method.md](proposed_method.md#objective-and-ranking).
The run starts with grounding/identity/coarse/state, then enables context after
the configured warmup. `train.epochs` is the total, including warmup;
`train.context_warmup_epochs` controls only context loss and must be smaller than
that total. Composition and fine context scoring are skipped during warmup. The
same optimizer continues, and the base objectives stay active afterwards.
Set `train.context_warmup_epochs=0` for a joint-training ablation or a one-epoch
smoke test. When context loss is disabled, all epochs use the remaining objectives.

Both optimizer groups use epoch-wise cosine decay from their configured base
rates to `optimizer.min_lr_ratio` times those rates (default `0.1`). Decay spans
all training epochs, independently of context and mining warmup. The first epoch
uses the base rates; the final epoch uses the floor. A one-epoch smoke test uses
the base rates. Actual rates are recorded in `history.jsonl` and W&B.

Periodic validation uses the same fine score and pooling as training. Coarse
scores select Top-M; final ranking uses the fine score directly. Changing
`retrieval.context_weight` or `retrieval.pooling` requires a newly trained run.

Training requires a fresh run directory. Obsolete loss/scoring configs and
checkpoints trained with a previous fine scorer are rejected. Frozen feature
caches can be reused when their provenance matches. To disable global-state
training, also set `retrieval.coarse_beta=0`.

Training uses only train annotations. Before epoch 1, GT heads are aligned to the
fixed detector crops required by train queries and their Full Positives. The
alignment is cached separately from visual features.

Each batch contains:

- up to `positives_per_query` Full Positives;
- `identity_fraction` same-identity, wrong-state negatives, with
  `ceil(same_ID_quota * semantic_hard_fraction)` selected from the semantic hard
  pool when it is available and the same-ID pool exceeds the quota;
- `hard_fraction` mined wrong/missing-ID negatives when context rejection is
  enabled, or the original coarse hard negatives when it is disabled;
- uniform negatives for the remaining slots.

Small same-ID pools are retained in full. Missing hard semantic evidence falls
back to random same-ID images first. Missing hard identity evidence falls back
to uniform negatives, without marking fallback images as rejection examples.
Self, all Full Positives (including unsampled ones), disputed negatives and
duplicate candidate IDs stay excluded. GT identity remains a crop-level label
for SupCon: a Full-negative image can still supply same-ID positive crops.

After `train.sampling.warmup_epochs` (default 8), fine mining refreshes every two
epochs. It first takes at most `pool_size` finite coarse candidates **per enabled
negative group** (same-ID/non-Full versus missing at least one required ID), then
fine-scores their union with the current model in eval/inference mode. The
shared bounded residual scorer ranks each candidate independently; mining
support differs from inference Top-M without changing the pairwise score.
Zero-coverage images do not enter fine hard pools.
Sampling draws from a hardest window of twice the requested hard quota, rather
than always taking the same top-K. No FAFA backbone is loaded for mining.

The default 32-candidate query with two positives and sufficient evidence has
2 hard semantic, 13 random same-ID, 3 hard identity and 12 uniform negatives.
`candidate_semantic_hard` and `candidate_identity_hard` report actual epoch counts,
not requested quotas. `hard_negatives.json` records the last refresh epoch,
retrieval settings, candidate support and per-query pools. The `coarse` pool
is retained only when the original hard sampler is in use.

Fine mining requires native FAFA query features for **train** as well as val;
their index/text compatibility is checked before label preparation and epoch 1.
If those queries are absent/stale, extend the existing query cache with:

```bash
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query
```

Completed detection, scene and FAFA image caches remain reusable. For controlled
ablations use a fresh `output.dir` for each run:

| Ablation | `train.sampling.semantic_hard_fraction` | `loss.context_rejection_weight` |
| --- | ---: | ---: |
| No semantic hard mining or rejection | 0 | 0 |
| Semantic hard mining only | 0.5 | 0 |
| Identity rejection only | 0 | 0.2 |
| Both (current YAML) | 0.1 | 0.2 |

Omitted new settings default to zero. Training optimizes
`L_context = L_semantic + context_rejection_weight * L_rejection` using the same
bounded residual fine score. Only mined wrong-ID candidates enter rejection,
and only when both the positive and negative retain predicted coverage. Each
component averages valid pairs per query and valid queries separately; empty
components return differentiable zero. Inference scores and checkpoint selection
are unchanged. All pool classification/loss masks use train annotations only.

The identity loss can additionally read up to
`train.identity_sampling.max_crops` detector-aligned train crops. These crops do
not enter the candidate set, FAFA scoring or state loss; they only keep the
supervised contrastive objective supplied with cross-image positives.
With `hard_negative_fraction=0.5`, after the existing sampling warmup, this same
budget also includes hard known-ID target crops mined inside the current batch.
Required-ID supplemental positives keep their baseline quota first; the remaining
budget is split between hard crops and the baseline random pool, with random
fallback when hard evidence is short. Unknown/multi-head crops are excluded from
this extra supervision. Fraction zero restores baseline sampling; image-level
`train.sampling.hard_fraction` retains its separate meaning. Epoch history and
W&B record `identity_hard_crops`, `identity_random_crops` (including sampled
positives), and `identity_supplemental_crops` as counts, not query averages.

The current model config sets `fine_matching.background_logit=-3.5`. This is
checkpoint-authoritative for train validation, `retrieve` and `run`; missing old
fields mean zero. Changing only an inference YAML does not override it. Use a new
training output and regenerate validation retrieval/calibration for a new
background. Feature caches do not need rebuilding. Test retains the background
frozen in validation selection; score replay cannot simulate a different one.

After context warmup, the model is evaluated on the complete validation split at
`evaluation.every_epochs` and on the final epoch. Warmup checkpoints are saved as
`last.pt` but do not compete for `best.pt`. The checkpoint selection formula stays

```text
0.5 * overall Full-mAP + 0.5 * macro-case Full-mAP.
```

Use a separate output directory for every seed or ablation:

```bash
python tools/run.py train --config configs/proposed.yaml \
  --set train.seed=1 output.dir=runs/proposed-seed1
python tools/run.py run --config runs/proposed-seed1/config.yaml --split val
```

### Seeds and reproducibility

`train.seed` fixes every random choice of a run. It seeds the Python, NumPy and
PyTorch generators once, which covers initialization and dropout. Each epoch
draws its batches and candidates from one generator and its identity crops from
another; both are derived from `(seed, stream, epoch)` with SHA-256, so no two
seeds, streams or epochs share a stream. `train.seed` must be a non-negative
integer.

`runtime.deterministic` (default `true`) also asks PyTorch for reproducible
kernels: `torch.use_deterministic_algorithms(True, warn_only=True)`, deterministic
cuDNN with benchmarking off, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` unless it is
already set. An operation without a deterministic kernel warns instead of
failing. Set it to `false` only to trade reproducibility for speed. `runtime.json`
records the seed and this flag.

Repeating a run with the same seed, YAML, caches, software stack and GPU model is
meant to give identical weights. Other GPUs or CUDA/PyTorch builds can still
differ in low-order bits, so compare seeds only within one environment. Hard
negative mining, validation, retrieval and calibration draw no random numbers.

Earlier revisions derived the epoch streams as `seed + epoch`, so seed 1 replayed
seed 0 shifted by one epoch, and they seeded neither NumPy nor Python and did not
request deterministic kernels. A run trained that way cannot be reproduced exactly
by this scheme: the same seed now samples different batches. Its checkpoint still
loads for retrieval.

W&B tracking is configured by the `wandb` block in `configs/proposed.yaml`.
Each run records only research configuration and cache provenance, not local
dataset paths. It logs losses, gradient norm and learning rates every
`log_every_steps`, the effective context weight and warmup flag, then epoch means
and validation metrics. `history.jsonl` also records the phase and effective
context weight. Set `mode: offline`
to sync later with `wandb sync`, or set `enabled: false` for untracked smoke
tests.

W&B config also records the effective `optimizer.schedule=cosine`. The existing
global gradient norm is accompanied by model/text-projection norms at the same
logging interval. Validation adds only ID-mAP, Full-mAP per populated query case,
and `val/candidate_recall`: the fraction of Full Positives retained in the
configured Coarse Top-M (recorded under `retrieval.top_m`). These validation
diagnostics are also stored in `history.jsonl`; checkpoint selection is unchanged.

## 4. Retrieve validation, then select and freeze it

```bash
python tools/run.py run --config configs/proposed.yaml --split val
python tools/run.py calibrate --config configs/calibration.yaml
```

Use the saved `runs/proposed/config.yaml` in the first command if training
used path or parameter overrides. List the run's `val/rankings.pt` under
`rankings` in `configs/calibration.yaml`. The artifact embeds its actual run
configuration, checkpoint fingerprint and fixed matching settings; no separate
model config or top-level `method` is needed in the calibration YAML.

Calibration selects among the listed validation artifacts. `coarse_beta` and
`top_m` define the shortlist of an artifact, so another value is compared by
retrieving validation again and listing the new artifact:

```bash
python tools/run.py run --config runs/proposed/config.yaml --split val \
  --set retrieval.coarse_beta=1.5 output.dir=runs/proposed-beta1.5
```

All artifacts must come from one checkpoint and the complete validation split.
The residual weight and pooling remain the trained values. The default
`selection_metric: checkpoint_score` matches training: the mean of overall
Full-mAP and macro-case Full-mAP over populated cases. `full_map` and
`macro_full_map` are also supported. Every artifact receives the official
evaluation; ties select the first listed artifact. `summary.csv` records
`macro_full_map` and `checkpoint_score`; `selection.json` records the criterion
and the difference to the first listed artifact.

Outputs are `summary.csv`, `selection.json`, `selected.yaml` and the best
`selected/val/{rankings.pt,run.json,metrics.json}`. Evaluation cutoffs are
`candidate_ks` together with `retrieval.top_m`, in training, `run`, `evaluate`
and calibration alike. CandidateRecall@K uses the complete coarse order.

Only the listed `rankings.pt` files and validation annotations are read. No
checkpoint weights, backbone, images or feature caches are opened. Each artifact's
checkpoint fingerprint and embedded selection guards must agree. Validation
fingerprints, complete query IDs, query indices and ordered gallery are checked
before writes; partial exports and test artifacts cannot be used to select
parameters. `selected.yaml` freezes all scoring settings and both
checkpoint/validation fingerprints. Test retrieval checks the real checkpoint
bytes against this hash. Matching thresholds, missing penalties and model
weights remain fixed.

`retrieval.pooling` accepts `penalized`, `matched_mean` and `subject_mean`.
The checked-in config uses `subject_mean`; omitted keys use `penalized`.
Every retrieval exports all three binding tensors for diagnostic replay. Deployable
retrieval and calibration retain the pooling and residual weight used in training.
Compare alternative values by training separate runs and selecting on validation.
The checked-in residual weight `0.1` is a starting value, not a measured optimum.
Monitor raw-context saturation when training the bounded reasoner output.

## 5. Run the frozen test configuration

```bash
python tools/run.py run \
  --config runs/proposed-calibration/selected.yaml \
  --split test
```

The runner verifies the checkpoint hash, validation fingerprint, feature cache
identity and encoder metadata before ranking. Test labels are used only by the
evaluator.

The selected validation result is already saved by calibration; it requires
no further model inference. Prepared caches and the checkpoint are needed again
only when retrieving the separate test split.

## Kaggle reuse

To resume a FAFA image-cache export from a Kaggle notebook, use the downloader
from the FAFA environment with explicit notebook coordinates:

```bash
.venv-fafa/bin/python download_fafa.py \
  --output cache/person-images --user OWNER --kernel NOTEBOOK --version VERSION
```

Authentication uses the normal Kaggle client configuration. The downloader
records its source in the resume state and refuses to continue that state with
different notebook arguments.
`--output` names the local destination. The downloader reads the historical
`cache/fafa-images` path inside the selected Kaggle export; that remote archive
prefix does not need to match the local `cache/person-images` directory.

The four cache paths include `data.detection_cache`. Completed unified CLIP scene
features are readable; historical boxes require conversion with the original
code revision or a fresh detection build. Build format-7 FAFA caches once, then they are independent of scene
backbones. Old DINO/BERT model checkpoints require fresh CLIP training. Keep
raw images available for the frozen FAFA crop extraction. Once new caches are
complete, mounted read-only copies are supported by training/retrieval.
See [feature contracts](feature_contracts.md) for text policy and checkpoint compatibility.
See [detection separation](detection_cache.md) for import and independent builds.

## Reading outputs

`history.jsonl` contains one compact row per epoch. Each validation directory
contains overall and by-case metrics. Split retrieval output additionally stores
coarse, fine, final and FAFA diagnostic rankings. The FAFA diagnostic reranks the
same coarse Top-M with FDA binding only; it is not the standalone FAFA baseline.
Checkpoints are inference/reproducibility artifacts, not resumable trainer states:
they omit optimizer/scaler state and store only the learned text projection,
native span policy and frozen CLIP revision.

For a fair ablation, change one owned component, keep data/evaluation fixed, and
record the commit, full YAML, seed, runtime and validation result.

## Score artifacts and CPU replay

`rankings.pt` is the sole score artifact. `SCORE_SCHEMA_VERSION`,
`score_schema_version` and `schema_version` are all **3**. Let Q be the number of
queries, G the split gallery size and M = min(requested Top-M, G-1).

| Key | Type/shape | Meaning |
| --- | --- | --- |
| `sample_ids`, `gallery_ids` | lists of strings | Ordered public identifiers |
| `query_indices` | int64 `[Q]` | Query image index in this ordered gallery |
| `coarse_topm` | int32 `[Q,M]` | Coarse prefix; column j of every `[Q,M]` tensor belongs to `coarse_topm[:, j]` |
| `fafa_binding`, `context`, `coverage` | float32 `[Q,M]` each | Raw fine evidence of the shortlisted pairs |
| `binding_penalized`, `binding_matched_mean`, `binding_subject_mean` | float32 `[Q,M]` each | FAFA reductions from the same assignment; `fafa_binding` selects one |
| `binding_scale` | float | Recorded alpha; fixed at 1 for new training runs |
| `state_trained` | bool | A false value forbids a positive beta at retrieval |
| `fine_matching` | dictionary | Fixed assignment thresholds, missing penalty and duplicate policy |
| `score_config` | dictionary | Pooling, fine mode, score weights and normalization/tie conventions |
| `provenance` | dictionary | Actual run config, checkpoint/cache fingerprints, benchmark and runtime metadata |
| `rankings`, `coarse_rankings`, `fine_rankings`, `fafa_rankings` | int32 `[Q,G-1]` each | Complete self-excluded diagnostic/final orders |
| `coarse_topm_scores`, `gated_context`, `coverage_penalty`, `context_residual`, `fine_scores` | float32 `[Q,M]` each | Derived scores for inspecting the shortlist |

Training validation, validation retrieval and test retrieval share one scoring
path: coarse scores order the whole gallery, and only the shortlist is scored by
the pairwise model, once per query/target in bounded `fine_batch_size` chunks.
Fine scores are computed independently per pair in FP32, with no shortlist
statistics. Coarse self scores and non-finite values are excluded from
normalization by the existing `combine_scores` helper. Fine scores remain finite,
including null matches. Stable coarse ties retain gallery order; shortlist ties
retain coarse order, and fine reranking preserves the coarse tail.

A run fine-scores `Q*M` pairs, about 0.9 million at 1,799 queries and M = 500.
The five evidence tensors occupy `20*Q*M` bytes, about 17 MiB at that size.
`fafa_binding` aliases the selected `binding_*` tensor, including after
saving/loading; treat score tensors as immutable.

Install the repository and CPU evaluation dependencies once. Replay then needs
only `rankings.pt`, without a checkpoint, GPU, encoder libraries or feature cache:

```bash
# Reproduce the recorded ranking exactly.
python -m rcr.proposed.replay runs/proposed/val/rankings.pt \
  --output-dir runs/replay-exact/val

# Diagnostic ablation: another pooling and residual weight on the saved shortlist.
python -m rcr.proposed.replay runs/proposed/val/rankings.pt \
  --output-dir runs/replayed/val --pooling matched_mean --context-weight 0.05

# Evaluate with the same fixed candidate_ks; no image/model inference.
python tools/run.py evaluate --config configs/proposed.yaml --split val \
  --set output.dir=runs/replayed
```

Omitted replay parameters reuse recorded values. The only fine scorer is
`bounded_residual`. `--pooling` and `--context-weight` act on fixed model
evidence: they do not retrain the reasoner, mark the output `fine_score_ablation`
and cannot be used as a calibration source. Official retrieval verifies both
against the training checkpoint. `--coarse-beta` and `--top-m` define the saved
shortlist, so replay accepts only their recorded values and reports any other
value as an error; retrieve validation again with the new value. Replay carries
source provenance into a separate directory and removes stale test selection
guards.

Replay requires the complete score-schema-3 contract; artifacts of an earlier
schema must be retrieved again. Changing model weights, FDA-k, matching rules,
detections, gallery, query text, coarse beta or Top-M requires new retrieval.
Only validation labels select parameters, and test labels are used solely for
final evaluation.
