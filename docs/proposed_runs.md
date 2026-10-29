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
The current YAML uses 6 warmup epochs for grounding/identity/coarse/state, followed
by 29 epochs with context enabled. `train.epochs` is the total, including warmup;
`train.context_warmup_epochs` controls only context loss and must be smaller than
that total. Composition and fine context scoring are skipped during warmup. The
same optimizer continues, and the base objectives stay active afterwards.
Set `train.context_warmup_epochs=0` for a joint-training ablation or a one-epoch
smoke test. When context loss is disabled, all epochs use the remaining objectives.

Periodic validation uses Context z-score with `context_weight=0.01` and
`fine_coarse_weight=0.05` in the current YAML, normalizing only after scoring a
query's complete Coarse Top-M. DINO/BERT checkpoints are incompatible; train a
fresh CLIP checkpoint. To reproduce the previous fine
ranking, use `--set retrieval.fine_score_mode=legacy_raw` at retrieval time.

Training requires a fresh run directory; obsolete loss configs with
`retrieval_weight` are rejected. Existing compatible checkpoints retain their
inference behavior, including when loaded through the previous architecture-name
alias. To disable global-state training, also set `retrieval.coarse_beta=0`.

Training uses only train annotations. Before epoch 1, GT heads are aligned to the
fixed detector crops required by train queries and their Full Positives. The
alignment is cached separately from visual features.

Each batch contains:

- up to `positives_per_query` Full Positives;
- `identity_fraction` of negative slots from same-identity, wrong-state images;
- `hard_fraction` of negative slots from coarse hard pools after
  `train.sampling.warmup_epochs`, independently of context warmup;
- uniform negatives for the remaining slots.

`train.sampling.identity_hard_fraction` reserves a share of the existing same-ID
quota for mined same-ID negatives **before** random same-ID sampling. The current
YAML uses `0.5`; an omitted setting resolves to zero. A value of `0.5` uses half
that quota when enough eligible mined images exist; shortages fall back to random
same-ID negatives, then uniform negatives.
The total candidate count and `identity_fraction` are unchanged. Mining starts
when either hard quota needs a pool, including when `hard_fraction=0`. Pools are
ranked by coarse score, not learned context score; this is an initial sampling
ablation, not context-error mining. Unreviewed positives can still be mislabeled
as negatives; only known positives and disputed pairs are excluded.

`optimizer.schedule` accepts `constant` or `cosine`. The current YAML uses cosine;
omitting the key retains a constant schedule. Cosine uses the
configured base rates in epoch 1 and `base * optimizer.min_lr_ratio` in the last
epoch, updating once at each epoch start. A single-epoch run uses base rates.
Both groups share the same factor; the optimizer and its state continue across
context/mining phases. With 25 epochs and ratio `0.1`, model LR is `1e-4 -> 1e-5`
and text projection LR is `1e-5 -> 1e-6`. This is decay, not LR warmup.

Step W&B logs and epoch history include realized candidate counts under
`sampled_positive`, `sampled_identity_hard`, `sampled_identity_random`,
`sampled_coarse_hard`, `sampled_uniform`, and `sampled_same_identity_negative`.
The last count overlaps the source quotas: uniform/general-hard draws may also
contain all required identities. Epoch logs additionally include `model_lr`,
`text_lr` and `mining_active`. Use these to check actual exposure, not just
configured fractions.

For the analyzed 25-epoch, seed-0 run, four complete configs are provided under
`configs/experiments/lr-negatives/`: `baseline.yaml`, `lr_only.yaml`,
`sampling_only.yaml`, and `combined.yaml`. All four retain the original run's
context warmup of 4 epochs, mining warmup of 6, candidates/seed/losses and fixed
original inference weights. They intentionally differ from today's
`configs/proposed.yaml`, which uses 35 epochs, 6 context warmup epochs and
10 mining warmup epochs. Its inference weights are starting settings. These
configs are candidate experiments, not validated improvements. Compare each single change with baseline before
interpreting the combined run; the old observed mAP is not a substitute for a
fresh control run when source/environment differ.

```bash
.venv-proposed/bin/python tools/run.py train \
  --config configs/experiments/lr-negatives/lr_only.yaml
.venv-proposed/bin/python tools/run.py train \
  --config configs/experiments/lr-negatives/sampling_only.yaml
```

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

## 4. Retrieve validation once, then calibrate offline

```bash
python tools/run.py run --config configs/proposed.yaml --split val
python tools/run.py calibrate --config configs/calibration.yaml
```

Use the saved `runs/proposed/config.yaml` in the first command if
training used path or parameter overrides. Set `rankings` in `configs/calibration.yaml`
to that run's `val/rankings.pt`. The artifact embeds its actual run configuration,
checkpoint fingerprint and fixed matching settings; no separate model config or
top-level `method` is needed in the calibration YAML.

Calibration jointly searches `coarse_beta`, `top_m`, `context_weight` and
`fine_coarse_weight`, using validation Full-mAP. The fine mode remains the one
recorded in the source artifact. Grid evaluation groups beta, Top-M, context
weight, then fusion weight to reuse coarse order, gathered raw scores,
context population statistics, fine diagnostics and fusion z-scores. The first
maximum in `summary.csv` wins ties. The cache retains only the current group.

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

The default context-weight grid includes 0 (context off), the initial 0.0325 and
a broader range up to 1 for new fixed-alpha training runs. These are validation
search points, not measured optima. Refine the grid on validation only.

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
`--output` names the local destination. `--cache-path` selects exactly one cache
directory inside the Kaggle export and defaults to `cache/person-images`.
For historical exports or their existing resume state, add
`--cache-path cache/fafa-images`. The remote directory and local destination may
differ; switching the remote directory requires a separate destination/resume state.

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
| `binding_scale` | float | Recorded alpha; fixed at 1 for new training runs |
| `state_trained` | bool | A false value forbids a positive beta during replay/calibration |
| `fine_matching` | dictionary | Fixed assignment thresholds, missing penalty and duplicate policy |
| `score_config` | dictionary | Fine mode, four selected parameters, normalization/alignment/tie conventions |
| `provenance` | dictionary | Actual run config, checkpoint/cache fingerprints, benchmark and runtime metadata |
| `rankings`, `coarse_rankings`, `fine_rankings`, `fafa_rankings` | int32 `[Q,G-1]` each | Complete self-excluded diagnostic/final orders |
| `coarse_topm` | int32 `[Q,M]` | Coarse prefix defining derived shortlist scores |
| `coarse_topm_scores`, `gated_context`, `fine_raw_scores`, `fine_scores`, `fused_topm_scores` | float32 `[Q,M]` each | Derived scores for inspecting the selected shortlist |
| `context_mean`, `context_std` | float32 `[Q]` each | Gated-context population statistics over Top-M |

Official validation retrieval always records the five full-gallery raw matrices.
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

The raw matrices occupy `20*Q*G` bytes, about 28.6 MiB at 264 queries and 5,684
images. Full-gallery collection requires `Q*(G-1)` fine pairs. The training
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
