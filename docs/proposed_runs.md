# Proposed v2 runs

Use the repository root. DINO/CLIP may need Hugging Face model access. Raw
images and pretrained weights are not distributed in this patch.

## Notebook setup

```python
import os
import sys

os.environ["PYTHON"] = sys.executable
```

Then run `!bash scripts/setup.bash proposed` in Colab/Kaggle. This installs the
complete proposed method, including its FAFA cache stage, into the kernel's
runtime: Torch 2.6.0, torchvision 0.21.0 and Transformers 4.57.6. Transformers
requires Torch >= 2.6 for the pinned CLIP revision's `pytorch_model.bin`;
that revision does not contain safetensors. There is no additional Python
path to configure. Install before model imports; restart the kernel after
upgrading an already loaded Torch runtime. Direct
commands below use the same Python (`{sys.executable}` in notebook shell cells).

## Build or reuse frozen features

Set `data.image_root`, `data.final_dir`, `data.dino_cache`, `data.cache` (FAFA),
`data.clip_cache` in `configs/proposed.yaml`.
`data.clip_cache` must be writable and different from both source directories.
The CLIP image/text model and processor share the pinned revision in this config.

```bash
# Prepare native FAFA only if compatible frozen person features are not ready.
python tools/run.py prepare --config configs/proposed.yaml

# Full build: reuse completed compatible stages; resume interrupted shards.
python tools/run.py build-cache --config configs/proposed.yaml

# Independent stages (the third needs completed DINO + FAFA sources).
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage dino
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage persons
python tools/run.py build-cache --config configs/proposed.yaml --cache-stage clip
```

Existing valid DINO/FAFA caches can stay in read-only Kaggle inputs. CLIP builds
only missing semantic features against exactly those detector boxes. If the DINO
cache lacks metadata, `cache.allow_legacy_dino=true` explicitly asserts that its
checkpoint/detector/preprocessing match. The example config enables this for
the existing published DINO cache. Use false and rebuild if the original
settings are unknown. Gallery order, patch grid and FAFA source bindings are
still checked. No GT boxes replace detector proposals.

The semantic cache contains `index.pt`, `features/<gallery-index>.pt`, `text.pt`,
`supervision.pt`, and the tokenizer. `.building` prevents partial caches from
being used. Text/annotation changes refresh their sidecars through the clip stage;
vision checkpoint/preprocessing/source mismatches require a new cache directory.
Model training/retrieval use cached tensors and do not import native FAFA.

### Notebook disk space

`OSError: [Errno 28] No space left on device` during Papermill autosave means
the output filesystem could not accept the notebook write. A subsequent
`NotJSONError` with an empty notebook is a consequence of that failed save.
The traceback alone does not identify which directory filled the disk.

Setup uses `--no-cache-dir` to avoid retaining a second copy of large pip wheels.
Cache/checkpoint/ranking writes check available bytes and reserve 1 GiB for
notebook saves and other small outputs. Failed writes remove their temporary
file and preserve the published file. Interrupted builds remove only known
cache temporary files after validating their build settings.

Before downloading CLIP weights or running gallery inference, the builder
estimates missing vision features, text replacement peaks and sidecars. Estimates
use the actual pinned model config and detected-person counts, including every
raw visual token. Compatible completed shards are reused. For the default
ViT-B/32 in FP16, the vision tensors alone take about 76 KiB per person;
the gallery size alone cannot predict the number of people or total cache size.
CLIP loading temporarily disables Transformers' optional background safetensors
conversion/download, which can retain an extra checkpoint. Normal format selection,
the pinned revision, safety checks and cache signatures are unchanged.

Check the notebook filesystem and pip cache before restarting a failed run:

```bash
df -h /kaggle/working
df -i /kaggle/working
du -xhd1 /kaggle/working /root/.cache 2>/dev/null || true
du -xhd1 cache checkpoints dataset runs 2>/dev/null || true
python -m pip cache info
# Clears downloaded pip artifacts; installed packages are retained.
python -m pip cache purge
```

Keep existing feature shards, build manifests and trained checkpoints.
The default proposed config enables `cache.release_fafa_assets`: after validating
complete compatible FAFA/DINO features, the CLIP stage removes only recorded
EVA/BLIP2 base weights and the native BERT cache from the dedicated FAFA runtime
cache. The tuned checkpoint, all features, other models and read-only assets stay.
The asset marker is removed so native preparation can restore them if needed.
Set this option false to retain base assets for another native FAFA use.
`prepare` skips native downloads while compatible complete FAFA features exist;
`prepare --force` still prepares native assets explicitly.

For a saved config from an earlier version, enable release when resuming:

```bash
%%bash
set -euo pipefail
python tools/run.py build-cache --config runs/proposed-v2/run_config.yaml \
  --cache-stage clip --set cache.release_fafa_assets=true
```

This frees downloadable model files, not feature shards. It leaves the cache's
raw tensors, precision, model revision and signatures unchanged. The actual
free-space change is printed before the CLIP estimate; enough capacity is still
required for the CLIP checkpoint download, later training checkpoints and rankings.

Use read-only inputs for already completed compatible caches and
pretrained weights instead of copying them into the writable output volume.
Keep paths/settings fixed when resuming an unfinished build. Once sufficient
space is available, rerun `build-cache` with the saved run config and the same
cache paths; completed stages will be reused. The 1 GiB reserve is a guard on
these repository writes, not a guarantee against other processes, model downloads
or filesystem quotas. It does not make an oversized full cache fit a smaller disk.
Restore a damaged output notebook from the editor or a valid earlier version;
an empty `.ipynb` contains no recoverable cells.

If a previous run stopped while loading CLIP, apply the runtime fix, rerun
setup and restart the kernel. Retain the existing cache directories: the model
revision and cache signatures are unchanged. To finish only that stage, using
the actual paths saved by the failed run:

```bash
%%bash
set -euo pipefail
python tools/run.py build-cache --config runs/proposed-v2/run_config.yaml \
  --cache-stage clip
```

Then run the val workflow below, followed by ablation after validation succeeds.
The val workflow reuses completed DINO/FAFA/CLIP caches. Keep `set -euo pipefail`
at the start of every notebook Bash cell that combines multiple commands so
ablation cannot continue after a failed cache or training stage. Use a new
training output directory if a prior completed epoch has already saved `last.pt`.

## Warmup, train, retrieve and evaluate

`bash scripts/methods/proposed.bash val` prepares required assets, builds/reuses all three
caches, trains/selects `best.pt`, and evaluates validation. It saves the exact
run config in `runs/proposed-v2/run_config.yaml`. A later
`bash scripts/methods/proposed.bash test` uses that config and selected checkpoint
without training or parameter selection. The val recipe includes training; use
the tools below to reuse a checkpoint or change configs/seeds.

```bash
# Default: 20 total epochs, including 2 grounding/identity warmup epochs.
python tools/run.py train --config configs/proposed.yaml

# Primary validation result: fine top-500 followed by coarse tail.
python tools/run.py retrieve --config configs/proposed.yaml --splits val
python tools/run.py evaluate --config configs/proposed.yaml --splits val

# Full-gallery fine control; separate output avoids overwriting primary rankings.
python tools/run.py retrieve --config configs/proposed.yaml --splits val \
  --set retrieval.mode=full output.dir=runs/proposed-v2-full
python tools/run.py evaluate --config configs/proposed.yaml --splits val \
  --set retrieval.mode=full output.dir=runs/proposed-v2-full

# Same trained checkpoint, fixed policies (full may be expensive).
# Suite base_config reads runs/proposed-v2/run_config.yaml from the val workflow.
python tools/run.py ablate --config configs/ablations/shortlist.yaml --splits val

# Use selected best.pt and fixed settings for test.
python tools/run.py run --config configs/proposed.yaml --splits test
python tools/report.py --split val --run proposed=runs/proposed-v2/val
```

If training uses a custom output directory, update the suite's `base_config`
to that directory's `run_config.yaml` (`config.yaml` for standalone `train`).
The saved config preserves the actual cache paths and legacy setting;
ablation does not reread the starter YAML.

`warmup.pt` is saved after warmup, `last.pt` after every epoch, and `best.pt` only
when main-training overall validation Full-mAP improves. Training requires a new
output directory if `last.pt` already exists; this version does not implement
optimizer resume. For a warmup-only diagnostic, set `train.epochs=2`,
`train.warmup_epochs=2` and a separate `output.dir`; it is not a retrieval model.

Kaggle T4 starting settings are batch size 2, 8 candidates, 8 training pairs per
chunk, fine batch 16, FP16 CUDA autocast and FP32 transport/attention regions.
These are conservative initial settings, not a measured T4 memory or speed claim.
Reduce batch/candidates if needed; long texts or many people increase memory.
There is no truncation or hidden cardinality cap. `model.max_subjects` bounds
textual role IDs explicitly and raises if exceeded.

`history.jsonl` includes loss terms, identity active-anchor rate, supervised
matching counts, sampling counts and maximum transport residuals/iterations.
With `train.amp=true`, an FP16 loss/gradient overflow retries the **same batch**
once in FP32 without loss scaling. GradScaler rejects the overflowing gradient
update and lowers its scale; only a finite, clipped retry updates AdamW. The
retry restores the pre-forward RNG state for dropout. `fp32_retries` in each
history row counts recovered batches, which are also printed during training.
Finite batches keep the normal AMP path. A nonfinite FP32 loss/gradient still
stops training before an optimizer update and records the epoch, sample IDs,
precision attempts and bad parameter names (for bad gradients) in
`nonfinite_batch.json`. Do not disable `error_if_nonfinite` or replace gradients
with zeros. Use `--set train.amp=false` for a full FP32 diagnostic if needed.

After applying a training fix, reuse the completed feature caches and start a
fresh training output; this does not resume the interrupted optimizer. In a
Kaggle `%%bash` cell, the **outer cell** must stop before ablation if train fails:

```bash
set -euo pipefail
python_bin="${PYTHON:-python}"
new_run="runs/proposed-stable-$(date +%Y%m%d-%H%M%S)"
"$python_bin" tools/run.py run --config runs/proposed-v2/run_config.yaml \
  --train --splits val --set "output.dir=$new_run"
"$python_bin" tools/run.py ablate --config configs/ablations/shortlist.yaml \
  --splits val --set "base_config=$new_run/run_config.yaml" \
  "output_dir=$new_run-policies"
```

This recovery requires all feature caches to be complete. It avoids repeating
setup/asset downloads/cache building, preserves earlier checkpoints and uses
the new validation-selected `best.pt` for all retrieval policies.

Per-epoch validation JSON contains overall, by-case and per-query metrics.
`run.json` records cache/checkpoint/version provenance, shortlist policy,
retrieval elapsed time and CUDA peak allocated memory where available.

## Diagnostics and tests

```bash
# Dataset/GT coverage diagnostic only; it does not select or alter retrieval scores.
python tools/data/audit.py --final-dir dataset/data/final \
  --cache cache/proposed-fafa --dino-cache cache/proposed-dino \
  --output runs/gt-coverage-diagnostic.json

# Small matching retrieve/evaluate subsets for environment smoke tests.
python tools/run.py retrieve --config configs/proposed.yaml --splits val --max-queries 8
python tools/run.py evaluate --config configs/proposed.yaml --splits val --max-queries 8

python -m pytest -q
python -m ruff check src tools tests labelstudio
```

Subset outputs are marked and are not full-split results. Run full retrieval
again before paper reporting. Oracle/GT coverage is a separate diagnostic file,
never a primary model input, score or checkpoint criterion.

The synthetic suite checks the transport optimum against an independent SLSQP
solution, gradients through active capacities, null/capacity/zero-mass behavior,
background versus unknown labels, long text/all mentions, role/member masks,
permutation invariance, no GT inputs, rank-gradient isolation, train/inference
score equality, coarse tail/self-exclusion, cache resume/reuse/provenance, and a
full disk-cache → warmup/train → mine → retrieve → evaluate smoke run. A tiny
same-identity condition task verifies that the reasoner can overfit synthetic
features. Optional real CLIP/DINO implementation tests use randomly initialized
small models, not downloaded pretrained weights. If the pinned FAFA source is
prepared, optional tests also import its native API, run image/composed
extraction with tiny backbones, and roundtrip a tiny Q-Former safetensors model
under the same Transformers runtime used by DINOv3.

The patch does not provide measured RCR mAP, trained v2 weights, full-backbone
integration results, T4 FP16 validation or CUDA latency/memory benchmarks. CPU
BF16 autocast tests are not a substitute for a T4 FP16 run. Native FAFA extraction
is exercised with a deterministic fake image-only trunk in the CPU suite.
