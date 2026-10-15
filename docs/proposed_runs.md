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

Alternatively, `PYTHON=python3.11 bash scripts/setup.bash proposed` and
`PYTHON=python3.11 bash scripts/setup.bash fafa` create the isolated environments,
including on notebook images without `ensurepip`. The host pip needs 22.3+.

Set `data.final_dir`, `data.image_root`, `data.dino_cache`, `data.cache`, and
`person_encoder.python` in `configs/proposed.yaml`. All relative paths are
relative to the repository root. Windows uses `.venv-fafa/Scripts/python.exe`.
For Kaggle, use venv Python paths explicitly in every cell.

## Main workflow

`bash scripts/methods/proposed.bash val` prepares assets, builds/reuses both
caches, trains, calibrates coefficients on validation and writes the selected
validation result to `runs/calibration/selected/val`. Run
`bash scripts/methods/proposed.bash test` later to use the frozen selected YAML.
The val recipe includes training. Use the individual tools below when reusing a
checkpoint, changing configs/seeds or running only one cache stage.

```bash
# Download only FAFA source, checkpoint and runtime assets needed by person encoding.
# Uses person_encoder.python; no baseline CLIP selector/detector is prepared.
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml

# Stream 1: DINO scenes/detections/crops; no FAFA environment required.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage dino

# Stream 2: FAFA persons from the completed DINO source.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage persons

# Or ensure both caches; compatible completed stages are skipped.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml

# Train from cached features, select best.pt, retrieve/evaluate complete val.
.venv-proposed/bin/python tools/run.py run --config configs/proposed.yaml --train --splits val

# Select both inference coefficients jointly on val; only that pair reaches test.
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml

# A later test run uses the same frozen selection.
.venv-proposed/bin/python tools/run.py run --config runs/calibration/selected.yaml --splits test
```

Equivalent one-command initial pipeline:

```bash
.venv-proposed/bin/python tools/run.py run --config configs/proposed.yaml --prepare --build-cache --train --splits val
```

For training another seed, reuse the cache and use a separate output directory.
`run --train` sets its checkpoint to that directory's `best.pt` automatically:

```bash
.venv-proposed/bin/python tools/run.py run --config configs/proposed.yaml --train --splits val \
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
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage persons
.venv-proposed/bin/python tools/run.py run --config configs/proposed.yaml --train --splits val
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
| `data.cache` | FAFA `index.pt` (persons + FP32 scene means) | References source ID; does not copy scene patches |

Both builders skip a completed compatible cache before loading models. A
completed cache with different encoders, gallery order or source ID fails
explicitly; choose a new output directory for new settings. DINO feature files are
checked when read, without scanning the entire directory on every reuse. A
missing DINO file raises an I/O error; it does not silently trigger a rebuild.
For FAFA, the completed `index.pt` owns the person features; per-image FAFA
shards are temporary extraction checkpoints and are removed after finalization.
The loader reads the same indexed person features for coarse and fine scoring.
`--force` remains an asset-preparation option.

Each unfinished directory has its own `.building` marker and `.build.pt`
manifest. Feature files and final indexes are published atomically. Repeating
the same stage preserves its cache ID and skips saved compatible images.
Different resume settings fail. Training/retrieval reject incomplete source or
person caches. After publishing the index, the marker is removed. An older
interrupted build without a resume manifest starts that incomplete stage again;
completed legacy DINO files remain untouched.

Large padded indexes are memory-mapped when loaded. FAFA index assembly uses a
padded tensor in CPU RAM and writes `index.pt.tmp` before publishing `index.pt`.
RAM must hold that padded tensor. The finalizer estimates disk needs plus a
margin; when necessary it removes already-consumed FAFA shards before writing.
If interrupted after such removal, resume re-extracts those images. DINO source
files are never deleted. FAFA stores FP32 scene means so later runs need not
repool all legacy scene files.

The default `all` stage builds/reuses DINO, then builds/reuses FAFA persons.
Native FAFA assets are loaded only by the person worker when extraction is
needed; there is no extra preflight subprocess. Use `prepare` first when assets
are missing. Complete caches can be reused without a FAFA environment or raw
images. The saved FAFA YAML is still needed by `build-cache` to compare extraction
settings.

The model uses architecture tag `shared-person-partial-match-v1`. Existing
checkpoints with that tag load with the same parameter keys and scores. Use the
updated method YAML. Regenerate earlier calibration YAMLs with the new
`calibrate` command because the old selector fields were removed. Earlier
architectures are rejected before loading caches/backbones. Train the current
model in a separate run directory; compatible DINO/FAFA caches remain reusable.
Source/cache IDs and both feature widths are checked against the checkpoint.

After a successful build, training/retrieval read cached features only. They do
not need native FAFA weights, its YAML or raw crop images. Training requires FAFA
person features when an external person-encoder config is supplied; retrieval compares
cache ID, feature widths and encoder metadata with the checkpoint. Encoder
environment paths are used only in prepare and cache extraction.

## Outputs and diagnostics

`runs/proposed-a` contains config, tokenizer, `last.pt`, `best.pt`, history,
training-data diagnostics and per-epoch train/val metrics. Split runs save
`rankings.pt`, `run.json`, `metrics.json`; `run` writes `summary.csv`.
Saved-result evaluation checks the recorded split fingerprint before writing
metrics, rejecting changed query text, labels, gallery records or head annotations.
Older results without a fingerprint retain the existing gallery/sample checks.
Use a new retrieval run to evaluate changed benchmark inputs.

W&B retains loss/metric keys and adds `train/binding_scale` and
`epoch/binding_scale`; fine coverage/member/empty-selection diagnostics are also
logged. Its cache config records both feature widths and encoder
provenance. It closes on exceptions. Nonfinite loss aborts before
backward/optimizer updates and writes `nonfinite_batch.json`. Model parameters,
geometry attention and losses stay FP32; CUDA training uses existing AMP/scaler.
Mining and evaluation use FP32. LRU, deduplication and one-batch prefetch remain.
Fine scores must be finite even when `fine_coarse_weight=0`; a failed reranker
cannot publish NaN-based rankings or enter validation calibration.

Training state loss needs eligible same-identity positive/negative pairs. A tiny
smoke run without them must set `loss.state_weight=0` and `retrieval.coarse_beta=0`.
Standalone query-limited retrieval/evaluation use matching `--max-queries`; normal
`run` evaluates complete requested query splits; calibration always uses complete val.

## Development

The method config describes the current architecture and its hyperparameters.
Edit the owning module directly (see [code structure](code_structure.md)); use a
separate branch and output directory for a new idea. Keep the commit/config/seed
and measurements, then retain successful changes in the main model.

`configs/calibration.yaml` selects only `coarse_beta` and `fine_coarse_weight` on
validation. It saves `selected.yaml`, `selection.json` and grid metrics. Ties keep
the first weight pair in YAML order. Selected checkpoint and validation hashes
must still match during later retrieval; test never selects parameters.
