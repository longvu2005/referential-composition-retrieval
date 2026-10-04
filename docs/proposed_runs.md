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
| `model`, `train`, `optimizer`, `loss` | Training |
| `retrieval`, `candidate_ks` | Periodic and standalone retrieval/evaluation |
| `evaluation` | Validation interval and fixed train subset size |
| `wandb` | Optional experiment tracking |
| `runtime.device` | Cache, training and inference |
| `checkpoint` | Standalone inference; `run --train` uses the new best.pt |
| `output.dir` | Root for model and split outputs |

`model` is loaded from the checkpoint during inference. A numeric
`retrieval.coarse_beta` overrides its beta; `null` inherits the checkpoint value.
The new method YAML uses per-query `coarse_normalization: zscore` and
`coarse_mode: identity_state`. `none` restores raw score fusion. Setting
`rerank: false` returns the complete coarse ranking, not just Top-M.
`candidate_ks` must not exceed `retrieval.top_m`.
All new training runs use the two Subject roles defined by the benchmark schema.

## Cache and checkpoint reuse

The detector is Grounding DINO; the frozen image encoder is DINOv3. Authenticate
with Hugging Face and accept the image checkpoint's license before the first
cache build. Defaults remain scene `[224,224]`, person `[256,128]` and FP16
storage; tensors are copied into compact CPU storage. Model computation loads
cached features as FP32. Cache format and learned parameter names are unchanged.

An existing person-based cache from the source commit can be reused. Keep
`index.pt` and `features/` together; the cache/gallery/build-ID checks remain.
Legacy caches without pooled global features are supported with a read-only
initial pooling pass. Changing detector/backbone/preprocessing requires a cache
rebuild. `run --build-cache` requires `--train` because rebuilding changes the
cache identity; standalone `build-cache` remains available.

Existing identity+state checkpoints from the source commit remain readable.
Keep the sibling `tokenizer/` directory. Old checkpoints that predate the state
branch still require retraining, as before.

## Training outputs

- `config.yaml`, `tokenizer/`: exact training settings and Subject tokens.
- `last.pt`: latest completed epoch, written atomically.
- `best.pt`: best full-val Full-mAP checkpoint, written atomically.
- `history.jsonl`: epoch losses and evaluation metrics, also saved with W&B off.
- `evaluation/epoch_NNN/{train,val}_metrics.json`: aggregate/by-case metrics.

A new `train` invocation starts from scratch and clears a previous best selection
and history in that output directory. Use a different `output.dir` for each
experiment. There is no resume flag. Training still samples one reviewed positive
and train-gallery negatives, groups batches by Subject count, and keeps the same
four losses. W&B records loss components, learning rates, gradient norm, identity
activity, grounding supervision and train/val metrics. It closes on exceptions.
Set `wandb.enabled=false` or `wandb.mode=offline` as needed.

`evaluation.enabled=false` produces `last.pt` only. Use standalone `train` for
this case; the `run --train` pipeline requires val to select best.pt.

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
Fine ranking uses only the fine score within Top-M; remaining images retain
coarse order. Equal scores now retain a stable canonical order.

## Ablations and beta selection

```bash
python tools/methods/run.py ablate --config configs/ablations/coarse.yaml
python tools/methods/run.py ablate --config configs/ablations/retrieval.yaml
python tools/methods/run.py ablate --config configs/ablations/loss.yaml
```

The default split is **val**. Coarse experiments reuse one checkpoint and raw
score computation across beta values. The sweep writes `selected.yaml` and
`selection.json`; retrieval ablations inherit that selection. Loss ablations
train separate checkpoints using the same schedule and seed. Existing frozen
caches are reused throughout.

Use the complete [ablation guide](ablations_vi.md) for test commands, metric
definitions, output paths and adding future experiments.
