# Run the proposed model

Run commands from the repository root. Python 3.11/3.12 is supported. DINO needs
new Transformers; native FAFA needs its pinned older version. Keep two venvs:

```bash
python3.11 -m venv .venv-proposed
.venv-proposed/bin/python -m pip install -r requirements/bootstrap.txt
.venv-proposed/bin/python -m pip install --no-build-isolation -r requirements/proposed.txt

python3.11 -m venv .venv-fafa
.venv-fafa/bin/python -m pip install -r requirements/bootstrap.txt
.venv-fafa/bin/python -m pip install --no-build-isolation -r requirements/fafa.txt
```

The proposed extra includes Torchvision, required by DINOv3's fast image
processor. Reinstall `requirements/proposed.txt` when updating an existing venv.

Set `data.final_dir`, `data.image_root`, `data.cache`, and
`person_encoder.python` in `configs/methods/proposed.yaml`. All relative paths are
relative to the repository root. Windows uses `.venv-fafa/Scripts/python.exe`.
For Kaggle, use venv Python paths explicitly in every cell.

## Main workflow

```bash
# Download only FAFA source, checkpoint and runtime assets needed by person encoding.
# Uses person_encoder.python; no baseline CLIP selector/detector is prepared.
.venv-proposed/bin/python tools/methods/run.py prepare --config configs/methods/proposed.yaml

# Check native FAFA assets/imports, then build scene/person features in two stages.
.venv-proposed/bin/python tools/methods/run.py build-cache --config configs/methods/proposed.yaml

# Train from cached features, select best.pt, retrieve/evaluate complete val.
.venv-proposed/bin/python tools/methods/run.py run --config configs/methods/proposed.yaml --train --splits val

# Select both inference coefficients jointly on val; only that pair reaches test.
.venv-proposed/bin/python tools/methods/run.py ablate --config configs/calibration/joint.yaml --splits val test

# A later test run uses the same frozen selection.
.venv-proposed/bin/python tools/methods/run.py run --config runs/calibration/selected.yaml --splits test
```

Equivalent one-command initial pipeline:

```bash
.venv-proposed/bin/python tools/methods/run.py run --config configs/methods/proposed.yaml --prepare --build-cache --train --splits val
```

For training another seed, reuse the cache and use a separate output directory.
`run --train` sets its checkpoint to that directory's `best.pt` automatically:

```bash
.venv-proposed/bin/python tools/methods/run.py run --config configs/methods/proposed.yaml --train --splits val \
  --set train.seed=1 output.dir=runs/proposed_seed1
```

## Cache and checkpoint migration

Old DINO caches/checkpoints are not FAFA/dual artifacts, even if their widths
match. Build the new `cache/proposed-fafa` and retrain. Do not overwrite useful
old run/cache directories. The loader rejects legacy formats, wrong encoder
specifications, and caches that differ from the training checkpoint.

Each cache stores source/preprocessing metadata, FAFA checkpoint SHA256, original
pixel boxes, scene boxes, compact pooled person features, DINO patches, identity
labels and CPU scene means. Scene/person widths are validated separately. The
`.building` marker stays present through both stages; interrupted FAFA extraction
cannot be consumed as a completed cache. Repeat the build command to rebuild;
this version does not add a general resumable detector/cache pipeline.

Before detection, a short FAFA worker checks its source, checkpoint, runtime
assets and native imports. Missing prerequisites abort before scene extraction.
The check does not load model weights or download assets; use `prepare` first.

After a successful build, training/retrieval read cached features only. They do
not need native FAFA weights loaded or raw crop images opened. Keep the FAFA YAML
for feature-provenance checks. Encoder environment paths are used only in prepare
and cache extraction.

## Outputs and diagnostics

`runs/proposed` contains config, tokenizer, `last.pt`, `best.pt`, history,
training-data diagnostics and per-epoch train/val metrics. Split runs save
`rankings.pt`, `run.json`, `metrics.json`; `run` writes `summary.csv`.

W&B retains loss/metric keys and adds `train/binding_scale` and
`epoch/binding_scale`; its cache config records both feature widths and encoder
provenance. It closes on exceptions. Nonfinite loss aborts before
backward/optimizer updates and writes `nonfinite_batch.json`. Model parameters,
geometry attention and losses stay FP32; CUDA training uses existing AMP/scaler.
Mining and evaluation use FP32. LRU, deduplication and one-batch prefetch remain.
Fine scores must be finite even when `fine_coarse_weight=0`; a failed reranker
cannot publish NaN-based rankings or enter validation calibration.

Training state loss needs eligible same-identity positive/negative pairs. A tiny
smoke run without them must set `loss.state_weight=0` and identity-only coarse.
Standalone query-limited retrieval/evaluation use matching `--max-queries`; normal
`run` and ablation suites evaluate complete requested query splits.

## New experiments

See [the ablation guide](ablations_vi.md). Calibration now lives in
`configs/calibration/joint.yaml`; the old coarse/loss/sampling/identity-amplitude
ablation configurations have been removed.
