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

Set `data.final_dir`, `data.image_root`, `data.dino_cache`,
`data.fafa_image_cache`, `data.fafa_query_cache`, and
`person_encoder.python` in `configs/proposed.yaml`. All relative paths are
relative to the repository root. Windows uses `.venv-fafa/Scripts/python.exe`.
For Kaggle, use venv Python paths explicitly in every cell.

## Main workflow

`bash scripts/methods/proposed.bash val` prepares assets, builds/reuses all three
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

# Stream 2: query-independent pooled persons and projected target tokens.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-image

# Stream 3: multimodal queries, updated incrementally when instructions change.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query

# Use --cache-stage fafa to run both FAFA stages with one model load.
# Or ensure all three caches; compatible completed stages are skipped.
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
  fafa_image_cache: /kaggle/working/cache/fafa-images
  fafa_query_cache: /kaggle/working/cache/fafa-queries-txt128
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
| `data.fafa_image_cache` | `index.pt`, `targets/` | Query-independent person means, projected target tokens and encoder/crop provenance |
| `data.fafa_query_cache` | `index.pt`, `features/<hash>.pt` | Separate sample/Subject mapping and FP32 text-conditioned embeddings |

New image indices use format 6: `persons` is `[N_persons, hidden_dim]` and
`person_offsets` is `[N_images + 1]`. Only actual people are stored; coarse batches
are padded when read. Completed dense format-4/5 indices remain readable without
rewriting them. The image signature has no queries or split membership.
Image-only pooled means and projected tokens share one Q-Former forward per crop.
The query index has its own cache ID; its signature binds the DINO source cache,
FAFA source/config/checkpoint hash, preprocessing and extraction precision. Each
query shard is keyed by that signature, reference image and public Subject
condition. Target and query crop order is inherited from the same DINO index.
No GT labels or grounding predictions enter native extraction.

Run `fafa-query` again after editing instructions. It reuses existing condition
shards, extracts only missing ones and atomically replaces the sample mapping.
Changing sample order does not change feature paths. Changing Subject order only
updates the mapping; identical image/condition pairs are extracted once. Old
condition shards stay available for reverting a text edit. If the query cache is
mounted read-only and instructions change, use a writable query directory; the
current builder does not overlay a writable layer on a read-only query cache.

Each unfinished FAFA cache has its own `.building` marker and `.build.pt`.
Repeating a stage skips valid saved shards and retains that cache's ID. Query edits
are allowed during either stage's resume. Changed weights, preprocessing, DINO
source ID or extraction precision require a new cache directory. Training/retrieval
reject incomplete caches; image-only readers are independent of query build state.

Pooled extraction shards in `.persons/` are temporary. They are assembled into
one compact, memory-mapped index and removed after publication. Finalization
holds only actual pooled person features, rather than a gallery-wide padded
tensor. Scene patches and global means are owned by DINO. Legacy global means
are pooled lazily on first use, without writing to the DINO source. Separate disk
checks reserve the image build's peak space and only missing query features,
including serialization overhead. Only validated reusable image shards reduce
the reservation. A format-5 interrupted image build can resume into format 6
with the same cache ID and valid shards. See [memory audit](cache_memory_audit.md).
Publish completed image features as
a Kaggle Dataset and point `data.fafa_image_cache` to its read-only mount to avoid
keeping that copy in `/kaggle/working`; query output can remain writable.

For a completed format-4 unified cache, replace `data.cache` with
`data.fafa_image_cache` pointing to the same directory and add a separate
`data.fafa_query_cache` path. The old image features, cache ID and native target
shards are reused without writes. Build `fafa-query` once; the old embedded query
files are not migrated. An unfinished old unified build has a different resume
signature: finish it using the old code or choose a new image cache directory.
Earlier person-only caches still require rebuilding. Two dependency environments
remain necessary because DINO and FAFA pin different Transformers versions.

Checkpoint writes check available disk space and use an atomic temporary file.
Failed writes remove that temporary and leave the previous checkpoint intact.
Training reserves estimated space for best, last and a temporary before epoch 1.
When best equals last, a hard link avoids another full copy; later replacement
of last preserves the older best. Filesystems without hard links use an atomic
copy. These checks report insufficient disk space; they cannot create more storage.

After a successful build, training/retrieval read cached features only. They do
not need native FAFA weights, its YAML or raw crop images. Training requires FAFA
person features when an external person-encoder config is supplied; retrieval compares
image cache ID, feature widths and encoder metadata with the checkpoint. Query
cache UUIDs are independent of checkpoints; the loader verifies matching encoder/crop
provenance and the current instruction mapping. Encoder
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
