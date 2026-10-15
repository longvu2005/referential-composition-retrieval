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

Set `data.final_dir`, `data.image_root`, `data.dino_cache`, `data.cache`,
and
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

# Stream 2: pooled persons + target tokens + multimodal queries in one FAFA cache.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa

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
  cache: /kaggle/working/cache/proposed-fafa-unified
```

Some notebooks mount the source at `/kaggle/input/rcr-proposed-cache` instead.
Keep your actual `final_dir`, raw-image root and worker Python paths. Raw images
are needed to extract FAFA crops; the DINO source supplies detections, so this
stage does not load DINO or rerun the detector:

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml
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
| `data.dino_cache` | DINO scenes, detections, identities, crop index | Shared immutable source; may be read-only Kaggle input |
| `data.cache` | `index.pt`, `targets/`, `queries/` | One FAFA cache ID, crop order and encoder provenance |

`index.pt` holds pooled person features, FP32 scene means and the native feature
manifest. Target shards hold projected Q-Former tokens; query shards hold FP32
multimodal embeddings for every Subject/crop. A single image-only forward supplies
both pooled means and target tokens. FAFA is loaded once per unfinished build.
Complete compatible caches are reused before loading FAFA or opening raw images.

Each unfinished cache has a `.building` marker and `.build.pt` signature. Repeating
the build skips valid saved image/query shards and retains its cache ID. Changed
weights, preprocessing, source ID or instructions require a new directory.
Training/retrieval reject incomplete caches. Query keys use public text and
Subject order; no GT labels or grounding predictions enter native extraction.

Pooled extraction shards in `.persons/` are temporary. They are assembled into
one memory-mapped padded index and removed after publication; target/query shards
remain final data. Scene patches are never copied from DINO. CPU RAM must hold
the padded person index during finalization. A disk estimate includes that index,
FAFA shards and serialization overhead before extraction starts.

Old separate FAFA caches are deliberately unsupported. Keep DINO, remove
`data.native_cache` from the YAML and build format 4 in a new `data.cache` directory.
Use a new run directory and recalibrate validation weights after retraining.
Two dependency environments remain necessary because DINO and FAFA pin different
Transformers versions; both use the same FAFA source/checkpoint loader.

Checkpoint writes check available disk space and use an atomic temporary file.
Failed writes remove that temporary and leave the previous checkpoint intact.
Training reserves estimated space for best, last and a temporary before epoch 1.
When best equals last, a hard link avoids another full copy; later replacement
of last preserves the older best. Filesystems without hard links use an atomic
copy. These checks report insufficient disk space; they cannot create more storage.

After a successful build, training/retrieval read cached features only. They do
not need native FAFA weights, its YAML or raw crop images. Training requires FAFA
person features when an external person-encoder config is supplied; retrieval compares
cache ID, feature widths and encoder metadata with the checkpoint. Encoder
environment paths are used only in prepare and cache extraction.

## Outputs and diagnostics

`runs/proposed-fafa-unified` contains config, tokenizer, `last.pt`, `best.pt`, history,
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
Edit the owning module directly (`cache/fafa.py`, `nn/`, losses or ranking); use a
separate branch and output directory for a new idea. Keep the commit/config/seed
and measurements, then retain successful changes in the main model.

`configs/calibration.yaml` selects only `coarse_beta` and `fine_coarse_weight` on
validation. It saves `selected.yaml`, `selection.json` and grid metrics. Ties keep
the first weight pair in YAML order. Selected checkpoint and validation hashes
must still match during later retrieval; test never selects parameters.

V1 also saves `fafa_rankings` and Top-M-aligned `fafa_binding`, `context`,
`coverage` tensors. `fafa_*` metrics use FDA with the same identity matching and
coarse shortlist, without contextual reranking. Compare them with `fine_*`,
final Full-mAP and Candidate Recall@500. To inspect all split candidates with
this diagnostic, set `retrieval.top_m` at least to the split gallery size minus
one; this can be much slower and still uses RCR identity matching/grounding.

`missing_penalty=1.1` is an initial FDA-scale setting, not a validation optimum.
The existing calibration command selects only coarse/fine fusion weights; it
does not tune penalty, thresholds or the learned binding scale. Choose matching
hyperparameters using train/val runs, then freeze them for test.
