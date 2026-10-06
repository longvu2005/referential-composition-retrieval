# Proposed: one config, explicit stages

All commands use `configs/methods/proposed.yaml`. Run from the repository root
with the proposed environment. Use `--set` or edit YAML to set your paths.

```bash
python tools/methods/run.py build-cache --config configs/methods/proposed.yaml
python tools/methods/run.py train --config configs/methods/proposed.yaml
python tools/methods/run.py retrieve --config configs/methods/proposed.yaml --splits val test
python tools/methods/run.py evaluate --config configs/methods/proposed.yaml --splits val test
```

Or use `run --build-cache --train` for the complete sequence. `run --train`
reuses the cache; `run` alone uses `checkpoint`. Stages run in separate Python
processes so model memory is released between cache building, training and
inference. The resolved config is saved as `run_config.yaml`.

## Config ownership

| Section | Used by |
| --- | --- |
| `data` | All stages, including the one shared cache path |
| `detector`, `image_encoder`, `cache.storage_dtype` | Cache building |
| `model`, `train`, `optimizer`, `loss`, `cache.prefetch_batches` | Training |
| `cache.lru_mib` | Training and inference feature reads |
| `retrieval`, `candidate_ks` | Periodic and standalone retrieval/evaluation |
| `evaluation` | Validation interval and fixed train subset size |
| `wandb` | Optional experiment tracking |
| `runtime.device` | Cache, training and inference |
| `checkpoint` | Standalone inference; `run --train` uses the new best.pt |
| `output.dir` | Root for model and split outputs |

`model` is loaded from the checkpoint during inference. A numeric
`retrieval.coarse_beta` overrides its beta; `null` inherits the checkpoint value.
The method YAML uses per-query `coarse_normalization: zscore` and
`coarse_mode: identity_state`. `none` restores raw score fusion. Setting
`rerank: false` returns the complete coarse ranking, not just Top-M.
`candidate_ks` must not exceed `retrieval.top_m`.
All new training runs use the two Subject roles defined by the benchmark schema.

## Cache and checkpoint reuse

The detector is Grounding DINO; the frozen image encoder is DINOv3. Authenticate
with Hugging Face and accept the image checkpoint's license before the first
cache build. Defaults remain scene `[224,224]`, person `[256,128]` and FP16
storage; tensors are copied into compact CPU storage. Batch collation loads
raw 768-dimensional features as FP32; training may then use CUDA AMP. The cache
format is unchanged. A shared trainable visual projection maps scene/person/global
features to `model.dim=384`; the frozen BERT has a separate 768-to-384 projection.
Identity/state dimensions are 128, attention uses 6 heads, FFN ratio is 2 and
attention/FFN dropout is 0.1. Only the text projection is in the text optimizer group.

An existing cache with the same detector, backbone and preprocessing can be reused. Keep
`index.pt` and `features/` together; the cache/gallery/build-ID checks remain.
Legacy caches without pooled global features are supported with a read-only
initial pooling pass. Changing detector/backbone/preprocessing requires a cache
rebuild. `run --build-cache` requires `--train` because rebuilding changes the
cache identity; standalone `build-cache` remains available.

Existing compatible identity+state checkpoints remain readable.
Keep the sibling `tokenizer/` directory. Old checkpoints that predate the state
branch still require retraining, as before.
Inference restores dimensions and dropout from the checkpoint's training config.
Train from scratch to use the new 384-dimensional model and frozen BERT policy;
editing the inference YAML does not resize an existing checkpoint.

## Training outputs

- `config.yaml`, `tokenizer/`: exact training settings and Subject tokens.
- `last.pt`: latest completed epoch, written atomically.
- `best.pt`: maximizes `0.5 * overall Full-mAP + 0.5 * macro-case Full-mAP`
  over full-val evaluations, written atomically; ties keep the earlier epoch.
- `history.jsonl`: epoch losses and evaluation metrics, also saved with W&B off.
- `evaluation/epoch_NNN/{train,val}_metrics.json`: aggregate/by-case metrics.
- `training_data.json`: resolved sampling settings and conflicting train labels.
- `hard_negatives.json`: latest train-only mined ID lists, when mining is enabled.

A new `train` invocation starts from scratch and clears a previous best selection
and history in that output directory. Use a different `output.dir` for each
experiment. There is no resume flag. Each epoch draws as many queries as the
train split contains, with replacement and case probability proportional to
`sqrt(N_case)`. The per-sample weight is `1/sqrt(N_case)`. Set
`train.case_balanced=false` for a shuffled pass without replacement. Draws are
grouped by Subject count, without dropping short batches.

Training samples up to two distinct reviewed positives among 24 candidates,
mixes same-identity, mined and random train-gallery negatives, and keeps the four
losses. The method YAML keeps `train.sampling.identity_fraction=0.5` and
`hard_fraction=0.3`: with two positives, 22 negative slots become 11 identity,
6 mined and 5 random slots. With only one positive, the remaining 23 slots become
11 identity, 6 mined and 6 random. Short pools fall back to random; all known
positives, the query image and disputed negatives remain excluded.
Mining starts after one completed epoch and refreshes
every two epochs. The pool is rebuilt for each new training run.

W&B and local epoch history record losses, identity activity, grounding
precision/recall on known aligned detections, any-match and complete-Subject
coverage, state supervised pair counts/active-query rate, and actual negative
source fractions and sampled positives per query. Validation history also records
`val/macro_full_map` (the equal mean over nonempty cases) and
`val/checkpoint_score`. W&B additionally records learning rates and gradient norm.
Grounding recall is conditional on valid detected/annotated people; use the cache
audit for missing-identity coverage. Loss summaries are weighted by query batch
size; count-derived rates use summed counts. W&B closes on exceptions.
Set `wandb.enabled=false` or `wandb.mode=offline` as needed.

`evaluation.enabled=false` produces `last.pt` only. Use standalone `train` for
this case; the `run --train` pipeline requires val to select best.pt.

State loss requires an explicit image-level identity mask. There is no fallback
that treats every candidate as a state negative. State-based validation fails
clearly if no supervised state pair has ever been seen. Use more identity
negatives or `retrieval.coarse_mode=identity_only` for an intentionally tiny
smoke run. When disabling `loss.state_weight`, also select identity-only
retrieval (or beta=0); the loss suite already does this.

## Training throughput options

The default YAML enables these options without changing the evaluation schedule,
Top-500 reranking, scoring/metrics, losses, optimizer settings or sampling quotas.
Validation still runs every two epochs and at the final epoch, with 100 fixed
train queries and the full val split, on each split's complete image gallery.

| Option | Default YAML | Behavior |
| --- | --- | --- |
| `train.mining_batch_size` | `8` | Group coarse mining queries by Subject count; omit unused fine composition. |
| `cache.lru_mib` | `1024` | Bound retained feature tensor storage in CPU RAM; `0` disables the LRU. |
| `cache.prefetch_batches` | `1` | Prepare one next CPU batch on a reader thread; `0` uses synchronous preparation. |
| `train.amp` | `true` | CUDA FP16 autocast and GradScaler during training; CPU falls back to FP32. |

Repeated image files are read once across query and candidate groups within a
batch. Their order, masks, supervision and separate query/target padding remain
intact. The LRU retains the cache's original storage precision, not expanded FP32
batches. Its budget covers cached tensor storage, not total process memory:
the person index, current/prefetched batches and Python objects also consume RAM.
Sampling and its RNG remain on the main thread; the worker only loads/pads CPU
data and pins batch memory on CUDA runs. BERT and GPU operations remain on the
main thread, and the worker is closed before mining or evaluation.

Only token IDs and Subject positions are cached, with dynamic batch padding.
Selection and change text pass through frozen BERT every step, then the trainable
text projection. BERT stays in eval mode, including when the wrapper enters train
mode, and runs without an autograd graph. No backbone parameter or Subject-token
embedding is optimized. All occurrences of each Subject marker remain available
to composition.

Mining preserves per-query normalization over the complete training gallery,
stable tie order and exclusion of queries, positives and disputed pairs. Gallery
projection is reused within each mining refresh, never across optimizer updates.
The sampler retains uniform sampling without replacement and the same fallback
quotas. Fixed seeds remain reproducible, but its new random-draw algorithm does
not reproduce the old `randperm` sample sequence bit for bit.

AMP keeps model parameters, optimizer state, geometry bias, normalization and
losses in FP32. Mining and evaluation run outside autocast in FP32. Logged
gradient norms are unscaled, and checkpoints include the scaler state (training
still starts from scratch; this does not add resume support). AMP can change the
numerical training trajectory, so retrieval quality still needs a real run.

Existing YAML files that omit these options retain conservative defaults:
mining batch 1, LRU 0, prefetch 0 and AMP off. To use them without replacing local
paths, add the four options above or pass them with `--set`. Reuse the existing
feature cache and the same training command; no cache rebuild is required.
For a synchronous FP32 comparison:

```bash
python tools/methods/run.py train --config configs/methods/proposed.yaml \
  --set train.amp=false train.mining_batch_size=1 \
        cache.lru_mib=0 cache.prefetch_batches=0 output.dir=runs/proposed_fp32
```

## Small smoke run

A separate copied smoke YAML is unnecessary. This still trains one complete
epoch over the train split with smaller batches; it is not a query-limited train:

```bash
python tools/methods/run.py train --config configs/methods/proposed.yaml \
  --set train.epochs=1 train.batch_size=2 train.candidates=4 \
        evaluation.enabled=false wandb.enabled=false output.dir=runs/smoke

python tools/methods/run.py retrieve --config configs/methods/proposed.yaml \
  --splits val --max-queries 2 \
  --set checkpoint=runs/smoke/last.pt output.dir=runs/smoke

python tools/methods/run.py evaluate --config configs/methods/proposed.yaml \
  --splits val --max-queries 2 --set output.dir=runs/smoke
```

Use the same `--max-queries` for retrieval and evaluation. Normal `run` always
uses the complete requested query splits. Train diagnostics search only train
images; val/test each search their own complete image split, excluding self.
Within Top-M, ranking uses the configured fine/coarse fusion
(`retrieval.fine_coarse_weight=0.4` in the method YAML); setting it to 0 restores
fine-only ranking. Remaining images retain coarse order. Equal scores retain
stable canonical gallery order.

## Joint 2D calibration and ablations

```bash
# Required calibration: select both coefficients together on val.
python tools/methods/run.py ablate --config configs/ablations/fine_coarse.yaml

# Optional diagnostics using that same checkpoint and selected pair.
python tools/methods/run.py ablate --config configs/ablations/coarse.yaml
python tools/methods/run.py ablate --config configs/ablations/retrieval.yaml

# Optional training ablations, each producing separate checkpoints.
python tools/methods/run.py ablate --config configs/ablations/loss.yaml
python tools/methods/run.py ablate --config configs/ablations/sampling.yaml

# Final test uses the joint selection unchanged.
python tools/methods/run.py run --config runs/ablations/fine_coarse/selected.yaml --splits test
```

The default split is **val**. `fine_coarse.yaml` reads the method YAML directly,
reuses one checkpoint, and evaluates the Cartesian product of 8 coarse beta
values and 7 fine/coarse weights. Both grids include zero and the existing
0.4/0.4 pair. The selection criterion is **final Full-mAP** over the complete
ranking, at fixed Top-500. CandidateRecall@500 is reported as a diagnostic;
it does not select beta in a separate first stage. Ties retain YAML grid order
(beta first, then fine/coarse weight).

Raw identity/state scores are computed once per query. Normalized coarse scores
and rankings are reused across final weights for each beta. Each query/target
fine score is computed once across the union of the beta-specific shortlists;
normalization for final fusion still uses each pair's own shortlist.

The sweep writes `summary.csv`, `selection.json` and `selected.yaml` under
`runs/ablations/fine_coarse/`. Both selected weights and checkpoint/validation
fingerprints are retained. Coarse and retrieval diagnostics inherit this
selection and never replace it. Existing sequential selections must be replaced
by a fresh joint sweep. `--splits val test` on the joint suite runs the full grid
on val and **only the winner** on test; a test-only suite invocation is rejected.
Use the selected method YAML for a later test-only run.

Loss ablations train separate checkpoints using the same schedule, seed and
identity-only retrieval, so untrained state projections never influence the
comparison. Sampling ablations compare random, identity/random and
identity/mined/random. Existing frozen caches are reused throughout. Selections
fingerprint val text/labels/gallery as well as the checkpoint; changing either
requires a new sweep. The method YAML keeps 0.4/0.4 for training-time validation;
post-training calibration does not rewrite checkpoint selection or training.

Use the complete [ablation guide](ablations_vi.md) for test commands, metric
definitions, output paths and adding future experiments.
