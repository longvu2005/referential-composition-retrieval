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

Set `data.final_dir`, `data.image_root`, `data.dino_cache`, `data.cache`, and
`person_encoder.python` in `configs/methods/proposed.yaml`. All relative paths are
relative to the repository root. Windows uses `.venv-fafa/Scripts/python.exe`.
For Kaggle, use venv Python paths explicitly in every cell.

## Main workflow

```bash
# Download only FAFA source, checkpoint and runtime assets needed by person encoding.
# Uses person_encoder.python; no baseline CLIP selector/detector is prepared.
.venv-proposed/bin/python tools/methods/run.py prepare --config configs/methods/proposed.yaml

# Stream 1: DINO scenes/detections/crops; no FAFA environment required.
.venv-proposed/bin/python tools/methods/run.py build-cache --config configs/methods/proposed.yaml --cache-stage dino

# Stream 2: FAFA persons from the completed DINO source.
.venv-proposed/bin/python tools/methods/run.py build-cache --config configs/methods/proposed.yaml --cache-stage persons

# Or ensure both caches; compatible completed stages are skipped.
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

## Reuse the existing Kaggle DINO cache

Add [rcr-proposed-cache](https://www.kaggle.com/datasets/phmhunhlongv/rcr-proposed-cache)
as a notebook input. Set these fields in the method YAML (use the actual mounted
directory containing `index.pt` and `features/`):

```yaml
data:
  final_dir: dataset/data/final
  image_root: /kaggle/input/YOUR_RAW_IMAGE_DATASET/images
  dino_cache: /kaggle/input/datasets/phmhunhlongv/rcr-proposed-cache
  cache: /kaggle/working/cache/proposed-fafa
```

Some notebooks mount the source at `/kaggle/input/rcr-proposed-cache` instead.
Keep your actual `final_dir`, raw-image root and worker Python paths. Raw images
are needed to extract FAFA crops; the DINO source supplies detections, so this
stage does not load DINO or rerun the detector:

```bash
.venv-proposed/bin/python tools/methods/run.py prepare --config configs/methods/proposed.yaml
.venv-proposed/bin/python tools/methods/run.py build-cache --config configs/methods/proposed.yaml --cache-stage persons
.venv-proposed/bin/python tools/methods/run.py run --config configs/methods/proposed.yaml --train --splits val
```

The supplied legacy cache has a 14 x 14 patch grid, 768 feature channels, FP16
storage and normalized scene boxes. It lacks encoder metadata and pixel boxes.
`cache.allow_legacy_dino=true` explicitly accepts that missing provenance; keep
the original DINOv3 ViT-B/16, 224 x 224 scene, 256 x 128 person letterbox and
Grounding DINO tiny settings. The loader checks the scene grid and gallery order.
Legacy pixel boxes are recovered by inverting the same letterbox transform using
current raw-image dimensions. This cannot verify missing encoder metadata. New
DINO caches store explicit metadata and pixel boxes.

## Cache lifecycle and checkpoints

| Directory | Contents | Usage |
| --- | --- | --- |
| `data.dino_cache` | DINO scenes, detections, identities, DINO crops/index | Shared immutable source; may be read-only Kaggle input |
| `data.cache` with FAFA | FAFA person features/index, FP32 global scene means | References source ID; does not copy scene patches |
| `data.cache` with DINO controls | The existing DINO directory | Reuses DINO crops/scenes without FAFA |

Both builders skip a completed compatible cache before loading models. A
completed cache with different encoders, gallery order, source ID or missing
feature files fails explicitly; choose a new output directory for new settings.
It is never silently replaced. `--force` remains an asset-preparation option.

Each unfinished directory has its own `.building` marker and `.build.pt`
manifest. Feature files and final indexes are published atomically. Repeating
the same stage preserves its cache ID and skips saved compatible images.
Different resume settings fail. Training/retrieval reject incomplete source or
person caches. After publishing the index, the marker is removed. An older
interrupted build without a resume manifest starts that incomplete stage again;
completed legacy DINO files remain untouched.

Large padded indexes are memory-mapped when loaded. FAFA index assembly uses a
temporary file-backed tensor, requiring temporary disk space approximately equal
to the final person index. FAFA stores FP32 scene means so later runs need not
repool all legacy scene files.

FAFA preflight runs only when new FAFA extraction is needed. The default `all`
stage checks prerequisites before building a missing DINO source; the DINO-only
stage is independent. Use `prepare` when assets are missing. Complete caches can
be reused without a FAFA environment or raw images.

Old DINO checkpoints still need new training for the FAFA/dual architecture.
Source/cache IDs and separate feature widths are checked against the training
checkpoint. Keep historical run directories separately.

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

See [the ablation guide](ablations.md). Calibration now lives in
`configs/calibration/joint.yaml`; the old coarse/loss/sampling/identity-amplitude
ablation configurations have been removed.
