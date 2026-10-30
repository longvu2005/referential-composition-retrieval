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

## 3. Train and select a checkpoint

```bash
python tools/run.py train --config configs/proposed.yaml
```

Training has separate grounding, crop-identity, coarse-identity, conditional
context and global-state objectives. The context loss ranks raw `coverage * context`
against same-ID negatives, without FDA or a trainable binding scale. Pairwise
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

Periodic validation uses the configured retrieval weights, normalizing context
after scoring a query's complete Coarse Top-M. `legacy_raw` remains available
through `retrieval.fine_score_mode` for compatible saved runs.

Training requires a fresh run directory; obsolete loss configs with
`retrieval_weight` are rejected. Existing compatible checkpoints retain their
inference behavior, including when loaded through the previous architecture-name
alias. To disable global-state training, also set `retrieval.coarse_beta=0`.

Training uses only train annotations. Before epoch 1, GT heads are aligned to the
fixed detector crops required by train queries and their Full Positives. The
alignment is cached separately from visual features.

Each batch contains:

- up to `positives_per_query` Full Positives;
- `identity_fraction` same-identity, wrong-state negatives;
- `hard_fraction` coarse hard negatives after `train.sampling.warmup_epochs`,
  independently of context warmup;
- uniform negatives for the remaining slots.

The identity loss can additionally read up to
`train.identity_sampling.max_crops` detector-aligned train crops. These crops do
not enter the candidate set, FAFA scoring or state loss; they only keep the
supervised contrastive objective supplied with cross-image positives.

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

## 4. Retrieve validation once, then calibrate offline

```bash
python tools/run.py run --config configs/proposed.yaml --split val
python tools/run.py calibrate --config configs/calibration.yaml
```

Use the saved `runs/proposed/config.yaml` in the first command if training
used path or parameter overrides. Set `rankings` in `configs/calibration.yaml`
to that run's `val/rankings.pt`. The artifact embeds its actual run configuration,
checkpoint fingerprint and fixed matching settings; no separate model config or
top-level `method` is needed in the calibration YAML.

Calibration jointly searches `pooling`, `coarse_beta`, `top_m`, `context_weight` and
`fine_coarse_weight`. The default `selection_metric: checkpoint_score` matches
training: the mean of overall Full-mAP and macro-case Full-mAP over populated
cases. `full_map` and `macro_full_map` are also supported. The fine mode remains
the one recorded in the source artifact. The exact source configuration is
evaluated first, even outside the grid, so the selected criterion cannot regress
against this baseline. Duplicate grid values are evaluated once.

Grid evaluation groups beta, Top-M, pooling,
context weight, then fusion weight to reuse coarse order, gathered raw scores,
context population statistics, fine diagnostics and fusion z-scores. The first
maximum in `summary.csv` wins ties, preserving the source on an equal score.
Labels are indexed once; AP/R@K use positive ranks with the official arithmetic.
Unchanged diagnostic branches reuse their metrics. The selected result is
evaluated again through the complete official evaluator before export is done.
The caches retain only the current group and one result per ranking branch.
`summary.csv` retains the existing metrics and adds `macro_full_map` and
`checkpoint_score`. `selection.json` records the criterion, baseline value and
improvement. Calibration still compares all poolings listed in its YAML; use
`pooling: [subject_mean]` to restrict the grid while retaining the source baseline.

Outputs are `summary.csv`, `selection.json`, `selected.yaml` and the best
`selected/val/{rankings.pt,run.json,metrics.json}`. Evaluation cutoffs
`candidate_ks` are preserved for every tuple and the selected config, even when
Top-M is smaller. CandidateRecall@K uses the complete coarse order.

Only `rankings.pt` and validation annotations are read. No checkpoint weights,
backbone, images or feature caches are opened. The artifact's checkpoint
fingerprint and embedded selection guards must agree. Validation fingerprints,
complete query IDs, query indices and ordered gallery are checked before writes;
partial exports and test artifacts cannot be used to select parameters.
`selected.yaml` freezes all scoring settings and both checkpoint/validation
fingerprints. Test retrieval checks the real checkpoint bytes against this hash.
Matching thresholds, missing penalties and model weights remain fixed.

`retrieval.pooling` accepts `penalized`, `matched_mean` and `subject_mean`.
The checked-in config uses `matched_mean`; omitted keys fall back to `penalized`.
These select only FAFA aggregation; `method`, checkpoint names,
model weights and cache formats stay unchanged. Validation exports all three
binding matrices from the same FDA forward pass, so calibration can sweep
`pooling` without model execution. If `pooling` is omitted from a calibration
YAML, the source artifact's pooling is retained. An older artifact can replay
its recorded pooling, but switching pooling requires a new validation export.

The context-weight grid includes context-off and larger weights. These are
validation search points, not measured optima. Refine the grid on validation only.

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
coarse, fine, fused and FAFA diagnostic rankings. The FAFA diagnostic reranks the
same coarse Top-M with FDA binding only; it is not the standalone FAFA baseline.
Checkpoints are inference/reproducibility artifacts, not resumable trainer states:
they omit optimizer/scaler state and store only the learned text projection,
native span policy and frozen CLIP revision.

For a fair ablation, change one owned component, keep data/evaluation fixed, and
record the commit, full YAML, seed, runtime and validation result.

## Score artifacts and CPU replay

`rankings.pt` is the sole score artifact. `SCORE_SCHEMA_VERSION`,
`score_schema_version` and `schema_version` are all **1**. Let Q be the number of
queries, G the split gallery size and M = min(requested Top-M, G-1).

| Key | Type/shape | Meaning |
| --- | --- | --- |
| `sample_ids`, `gallery_ids` | lists of strings | Ordered public identifiers |
| `query_indices` | int64 `[Q]` | Query image index in this ordered gallery |
| `identity_scores`, `state_scores` | float32 `[Q,G]` each | Raw coarse branches, including state when original beta is zero |
| `fafa_binding`, `context`, `coverage` | float32 `[Q,G]` each | Raw fine evidence for all non-self pairs |
| `binding_penalized`, `binding_matched_mean`, `binding_subject_mean` | float32 `[Q,G]` each | FAFA reductions from the same assignment; `fafa_binding` selects one |
| `binding_scale` | float | Recorded alpha; fixed at 1 for new training runs |
| `state_trained` | bool | A false value forbids a positive beta during replay/calibration |
| `fine_matching` | dictionary | Fixed assignment thresholds, missing penalty and duplicate policy |
| `score_config` | dictionary | Pooling, fine mode, score weights and normalization/alignment/tie conventions |
| `provenance` | dictionary | Actual run config, checkpoint/cache fingerprints, benchmark and runtime metadata |
| `rankings`, `coarse_rankings`, `fine_rankings`, `fafa_rankings` | int32 `[Q,G-1]` each | Complete self-excluded diagnostic/final orders |
| `coarse_topm` | int32 `[Q,M]` | Coarse prefix defining derived shortlist scores |
| `coarse_topm_scores`, `gated_context`, `fine_raw_scores`, `fine_scores`, `fused_topm_scores` | float32 `[Q,M]` each | Derived scores for inspecting the selected shortlist |
| `context_mean`, `context_std` | float32 `[Q]` each | Gated-context population statistics over Top-M |

Official validation retrieval records all raw branches and FAFA pooling matrices.
There is no separate tuning payload or opt-in flag. Raw fine values are stored
once in gallery order; derived shortlist scores keep coarse Top-M order. Raw fine
self entries are zero sentinels. Coarse self scores and non-finite values are
excluded from normalization by the existing `combine_scores` helper. Positive
fine/coarse fusion masks candidates with non-finite coarse scores as before.

The direct shortlist is scored first, followed by the remaining coarse tail,
once per query/target and in bounded `fine_batch_size` chunks. Population
statistics are computed on CPU float32 over the complete relevant vector, never
per inference chunk. Stable coarse ties retain gallery order; shortlist ties
retain coarse order, and fine reranking preserves the coarse tail.

The seven distinct raw matrices occupy `28*Q*G` bytes, about 40.1 MiB at 264
queries and 5,684 images. `fafa_binding` aliases the selected `binding_*` matrix,
including after saving/loading; treat raw score matrices as immutable.
Full-gallery collection requires `Q*(G-1)` fine pairs. The training
validation loop and test retrieval only compute Top-M fine pairs, and record
`score_alignment: coarse_topm`; their shortlist artifacts cannot supply offline
calibration. Official validation records `score_alignment: gallery_ids`.
No scoring formula, matching, architecture, training or evaluation target set
changes with this storage policy.

Install the repository and CPU evaluation dependencies once. Replay then needs
only `rankings.pt`, without a checkpoint, GPU, encoder libraries or feature cache:

```bash
# Reproduce the recorded ranking exactly.
python -m rcr.proposed.replay runs/proposed/val/rankings.pt \
  --output-dir runs/replay-exact/val

# Rebuild coarse order, choose another Top-M, normalize and fuse offline.
python -m rcr.proposed.replay runs/proposed/val/rankings.pt \
  --output-dir runs/replayed/val --coarse-beta 0.2 --top-m 500 \
  --context-weight 0.0288 --fine-coarse-weight 0.0064

# Evaluate with the same fixed candidate_ks; no image/model inference.
python tools/run.py evaluate --config configs/proposed.yaml --split val \
  --set output.dir=runs/replayed
```

Omitted replay parameters reuse recorded values. The existing `legacy_raw`,
`raw_gamma`, `context_zscore` and `both_zscore` fine modes remain available through
`--mode`; `--fine-coarse-weight 0` yields fine-only ordering. Replay carries the
source hash/provenance into a separate directory and retains every raw matrix
for successive replays, including after reducing Top-M. It removes stale test
selection guards from replayed config metadata. Use calibration to generate a
new frozen `selected.yaml`.

Replay requires the complete version-1 raw-score contract. Missing evidence must
be collected by running official validation retrieval again; it is never
fabricated from a saved shortlist. Changing model weights, FDA-k, matching rules,
detections, gallery or query text requires new retrieval. Only validation labels
select parameters, and test labels are used solely for final evaluation.
