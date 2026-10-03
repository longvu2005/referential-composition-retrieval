# CLIP and FAFA on RCR

One checkout, one final dataset, one official evaluator. Separate Python
environments isolate each solution's libraries; they do not create separate
benchmark protocols. Run commands from the repository root.

## Environments

Use Python 3.11 for new environments. Python 3.12 is also within the package's
supported range. FAFA is verified here on Linux x86-64;
use a Linux GPU server or Kaggle for that solution. CLIP/proposed development
can use the local platform supported by their PyTorch build.

```bash
python3.11 -m venv .venv-proposed
python3.11 -m venv .venv-clip
python3.11 -m venv .venv-fafa

.venv-proposed/bin/python -m pip install -r requirements/bootstrap.txt
.venv-clip/bin/python -m pip install -r requirements/bootstrap.txt
.venv-fafa/bin/python -m pip install -r requirements/bootstrap.txt
```

Before installing the solution requirements, install a PyTorch build appropriate
for your GPU/driver. For FAFA the declared pair is `torch==2.5.1` and
`torchvision==0.20.1`; for example, on an NVIDIA machine with a driver compatible
with CUDA 12.1:

```bash
.venv-fafa/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Use the [official PyTorch installer](https://pytorch.org/get-started/locally/)
or [previous-version instructions](https://pytorch.org/get-started/previous-versions/)
for other compute platforms. CPU verification can use the CPU wheel index;
the full FAFA model and gallery encoding are intended for GPU runs.

```bash
.venv-proposed/bin/python -m pip install --no-build-isolation -r requirements/proposed.txt
.venv-clip/bin/python -m pip install --no-build-isolation -r requirements/clip.txt
.venv-fafa/bin/python -m pip install --no-build-isolation -r requirements/fafa.txt

.venv-proposed/bin/python -m pip check
.venv-clip/bin/python -m pip check
.venv-fafa/bin/python -m pip check
```

The bootstrap file pins build tooling for the old OpenAI CLIP `setup.py` that
imports `pkg_resources`. `--no-build-isolation` makes that build use the prepared
setuptools instead of an isolated newer version. All runtime dependencies are
in `pyproject.toml`; the requirements files select
extras without duplicating those lists. Do not combine `proposed` and `fafa`
extras: their Transformers requirements intentionally conflict.

FAFA declares [eva-decord 0.6.1](https://pypi.org/project/eva-decord/0.6.1/), which
provides the same `decord` import API with correct Python-3 wheel metadata. The
original decord 0.6.0 Linux wheel embeds a CPython-3.6 tag and causes `pip check`
to reject it ([upstream issue](https://github.com/dmlc/decord/issues/356)). The
official FAFA image retrieval model/scoring is unchanged; video decoding is not
used by this RCR image adapter. Use a clean FAFA environment so two packages do
not compete for the `decord` namespace.

The `dataset` extra is installed with proposed for existing dataset tools;
loading an already finalized dataset does not need those construction tools.
`requirements.txt` remains a proposed+dataset installation selector for older
workflows. No inference script installs packages.

On Windows, create the environments with `py -3.11 -m venv ...` and replace
`bin/python` with `Scripts/python.exe`.

On Kaggle, if `venv` fails at `ensurepip`, create the environment without pip
and let the notebook's existing pip bootstrap it (pip 22.3 or newer):

```bash
python3 -m venv --without-pip .venv-clip
python3 -m pip --python .venv-clip install -r requirements/bootstrap.txt
```

Repeat with `.venv-fafa` or `.venv-proposed` as needed, then install the solution
requirements as above. The bootstrap file declares `wrapt` for startup hooks such
as `sitecustomize`; this is separate from model dependencies.
Call the chosen environment's Python through `subprocess.run([...], check=True)`
or activate it and run commands together within one `%%bash` cell. Shell
activation does not change the notebook kernel. Environments can be rebuilt each
session while images, checkpoints, feature caches and results are restored
separately.

## Data and preparation

Set `data.image_root` in both baseline YAML files to the PIPA image directory.
Images below that root retain the finalized relative paths, such as
`val/<image>.jpg`. Use the same `data.final_dir` as proposed. The repository does
not include those raw images or model weights.

Prepare artifacts explicitly before inference:

```bash
.venv-clip/bin/python tools/methods/prepare_baseline.py \
  --config configs/methods/baselines/clip.yaml

.venv-fafa/bin/python tools/methods/prepare_baseline.py \
  --config configs/methods/baselines/fafa.yaml
```

CLIP preparation downloads the named official checkpoint with its SHA-256.
FAFA preparation pins the authors' Git source, downloads the released checkpoint,
CLIP reference-selector weights and Faster R-CNN weights, then constructs the
official model on CPU once to populate its required runtime caches. This phase
needs internet and sufficient CPU RAM/disk for the official model assets.
Its runtime marker records the source/model and asset inventory.

Google Drive share URLs are converted to direct `uc?id=...` URLs before calling
gdown. Preparation does not use the removed `fuzzy` argument and supports the
declared gdown 5.2–6.x range.

Inference checks the source commit, real tracked edits and asset inventory before
expensive detection. Official imports disable bytecode writes. The FAFA model
loads from prepared assets with networking blocked; CLIP and the detector also
load explicit local checkpoints. Use `--force` in preparation only when replacing
artifacts is intended. Existing local checkpoints and valid caches can be reused.

## One-command experiments (recommended)

```bash
# Prepare once, run all four CLIP baselines on val and test, and evaluate.
bash scripts/run_baselines.bash clip --prepare
# Subsequent runs reuse the checkpoint and caches.
bash scripts/run_baselines.bash clip
# FAFA runs the same retrieval + evaluation protocol in its own environment.
bash scripts/run_baselines.bash fafa --prepare
```

The wrapper uses `.venv-clip/bin/python` or `.venv-fafa/bin/python`. Override with
`BASELINE_PYTHON=/path/to/python`; `--config path/to/config.yaml` is forwarded to
both preparation and the runner. No package installation occurs in the scripts.
For example, in one Kaggle `%%bash` cell, first `cd` to the repository root, then
run the same commands. After installing `requirements/clip.txt`, all four CLIP
baselines need **one local OpenAI CLIP checkpoint**, which contains the pretrained
image and text encoder weights. Fusion requires no additional model/weights file,
no detector and no RCR fine-tuning.

```bash
# Select each fusion's weights on full val and evaluate val.
bash scripts/run_baselines.bash clip --splits val
# Reuse exactly that saved selection; this command never tunes on test.
bash scripts/run_baselines.bash clip --splits test
# Subsets of methods are also supported.
bash scripts/run_baselines.bash clip --modes clip_text clip_image
bash scripts/run_baselines.bash clip --modes early_fusion late_fusion
```

The direct Python entry point is also supported (including Windows):

```bash
.venv-clip/bin/python tools/methods/run_baselines.py \
  --config configs/methods/baselines/clip.yaml --splits val test
```

`run_baselines.py` defaults to all four modes and val/test. FAFA uses its own YAML
with the same CLI, without `--modes`. The runner always processes val first,
regardless of the order passed to `--splits`. Output directories must distinguish
modes and splits. `summary.csv` reports only the current invocation; per-split
`metrics.json` files persist until that split is run again.

### Fusion selection

`tuning.image_weights` is the alpha grid (default 0 to 1 in steps of 0.05):
`image_weight = alpha`, `text_weight = 1 - alpha`. Only the ratio matters, so two
independently searched weights are unnecessary. Each fusion mode selects its own
alpha, maximizing `tuning.metric` (default `full_map`) on **all validation queries**
with the official evaluator. Exact ties prefer the alpha closest to 0.5, then the
smaller alpha. Endpoints include the image-only and text-only rankings.

`runs/clip/tuning.json` stores every trial, the selected weights, metric, query
count and validation/config fingerprint. It is saved before test scoring begins.
Each fusion's `run.json` records the actual selected weights and val provenance.
The test-only command requires a compatible saved selection; changes to the
checkpoint, precision, text field, normalization, grid, val labels or val image
fingerprints require a new val run. Keep the val inputs available for this check.
Image fingerprints include absolute paths, sizes and mtimes, so relocating or
restoring images can also require rerunning val. Test labels never enter weight
selection or its fingerprint. A val run replaces the selection with the requested
fusion modes; select both if both will later be evaluated on test.

This is a grid optimum on validation, **not a guarantee of the best test score**.
Decide the metric/grid before evaluating test. Validation results for fusion are
tuning results; test remains the held-out report. The current checked-in dataset
contains 264 val queries and only 14 test queries, so test is still small.

### Low-level retrieval / evaluation

The existing single-run commands remain available. `image`/`text` remain legacy
aliases with their existing directory names; new experiment runs use
`clip_image`/`clip_text`. Match the mode and split in both commands:

```bash
.venv-clip/bin/python tools/methods/retrieve_baseline.py \
  --config configs/methods/baselines/clip.yaml --mode clip_text --split val
.venv-clip/bin/python tools/methods/evaluate.py \
  --config configs/methods/baselines/clip.yaml --mode clip_text --split val

.venv-fafa/bin/python tools/methods/retrieve_baseline.py \
  --config configs/methods/baselines/fafa.yaml --split val
.venv-fafa/bin/python tools/methods/evaluate.py \
  --config configs/methods/baselines/fafa.yaml --split val
```

Low-level fusion retrieval uses the explicit `fusion` weights in YAML; it does
not select or load tuned weights. Use `run_baselines.py` for the val-to-test
selection protocol. For smoke checks, add matching `--max-queries 4` to low-level
retrieve/evaluate commands; the gallery remains complete. The experiment runner
intentionally uses complete splits so smoke subsets cannot become tuned results.
A new retrieval removes stale metrics before saving its new rankings.

Outputs per method/mode/split:

```text
scores.npy      float32 [num_queries, num_split_gallery], higher is better
rankings.pt     sample_ids, gallery_ids, int32 rankings [Q, G-1]
run.json        resolved config, versions, checkpoint hash, cache IDs, diagnostics
metrics.json    official evaluator output (runner writes this automatically)
```

Raw scores include the self-image; shared `scores_to_rankings` removes it before
saving the complete ranking. Ties use canonical gallery order. The evaluator
checks exact query/gallery order, no duplicates, no self-image and full coverage.
It computes unchanged `ID-mAP/R@1/5/10`, `Full-mAP/R@1/5/10`, per-case and per-query
metrics. `R@K` means at least one positive in top K. `coarse_topm` and
`CandidateRecall@K` remain optional proposed-stage diagnostics.

## Method policies and source

Scoring is ported from
[cpr_baseline_bench at ca774cc1](https://github.com/longvu2005/cpr_baseline_bench/tree/ca774cc1fe321b00d8c638975deba4cbdbaaf035).
Its CPR data preparation, query/gallery schema, evaluator and old metrics are
not used. All methods load `rcr.methods.common.data` and the same RCR split gallery.

| CLIP mode | Query | Score |
|---|---|---|
| clip_image | Normalized whole-scene image embedding | Image dot gallery image |
| clip_text | Normalized `final_instruction` embedding | Text dot gallery image |
| early_fusion | Normalized weighted sum of image/text embeddings | Fused query dot gallery image |
| late_fusion | Image/text branch scores | Per-query population z-score of each branch, weighted sum |

CLIP uses the pinned OpenAI implementation and `ViT-L/14`. The low-level
fixed-weight config starts at 0.5/0.5; the experiment runner selects on val.
Alternative `final_change` text is an explicit config variant. Truncated text
counts are recorded; no LLM rewriting occurs. Late-fusion z-score
statistics use the full split gallery, including self, as in the existing baseline;
the shared ranking step then removes self. This policy is identical during tuning
and final evaluation.

Image features, text features and branch score matrices are reused across modes.
For normalized embeddings, early fusion computes
`(wi * image_scores + wt * text_scores) / ||wi * image_query + wt * text_query||`;
this is algebraically the same as the original fused-embedding dot product.
Late fusion mixes the two per-query z-scored branches. Small floating-point
rounding differences from the direct early-fusion matrix product are possible.
No encoder or query-gallery matrix product is repeated for each alpha.
Score arrays are memory-mapped and mixed in batches on CPU; CLIP encoding and
initial branch matrix products use the configured device. Caches may be rebuilt
once because this patch versions the CLIP cache schema. A warm cache still checks
the checkpoint hash and inputs but does not load the encoder.

FAFA imports
[the official source at 0cc16936](https://github.com/Delong-liu-bupt/Composed_Person_Retrieval/tree/0cc16936f031f7ad166be4cce1be33d0b44b728e/FAFA_SynCPR)
and uses its released `tuned_recall_at1_step.pt`, `blip2_fafa_cpr/pretrain`,
squarepad test transform, multimodal query features and target token features.
Checkpoint loading uses the authors' `strict=False` policy and records every
missing/unexpected key. Soft FDA is the mean of the top six query-to-target-token
similarities, matching the old adapter and official inference.

The RCR adaptation is explicitly **FAFA + SetMatch with predicted Subject sets**:

1. Faster R-CNN predicts scene person candidates. Empty detections have a logged
   whole-scene fallback; no GT person/head boxes enter retrieval.
2. CLIP ViT-B/32 aligns `final_desc` Subject descriptions to candidate crops.
   Hungarian assignment gives one distinct anchor per Subject. INDIVIDUAL keeps
   one member. Other cases can include unused crops passing the absolute
   threshold and relative-to-best margin; competing Subjects assign a crop to
   the highest-scoring eligible Subject. The per-Subject cap is a hyperparameter.
3. Subject changes are split only at explicit DUAL Subject clause boundaries.
   Otherwise the intact condition is preserved per component. Ordered relational
   mentions remain in text. Each predicted member is encoded independently.
4. For each gallery scene, native FAFA component-person scores enter maximum-sum
   one-to-one Hungarian assignment. The final image score is the minimum assigned
   score, with unmatched components padded at -1. One component reduces to max
   person score. This is not maximum-minimum assignment.

The selector reads public case structure/Subject IDs, reference image and final
text only. It never reads `identity_ids`, their length, `target_image_id`, positives
or GT box mappings for scoring/selection. Gallery identities are evaluator labels.
GROUP and multi-member DUAL/RELATIONAL membership is a prediction, not an oracle.
Threshold 0.20/margin 0.03/cap 5 are initial unvalidated settings, logged in config.

Independent FAFA components approximate collective conditions and relations;
they do not become joint group/relation reasoners. Report the method as an RCR
adaptation and include these limits and predicted-member/fallback diagnostics.
Detector/backbone/supervision differ from proposed and should be reported in the
comparison table. Fine-tuning FAFA on RCR would be a separate experiment.

## Verification and cache reuse

CPU tests exercise the real RCR loader, feature scoring, caches, ranking output,
CLI and official metrics using deterministic small encoders. They also check
native FDA, max-sum-then-min matching, predicted multi-member selection and that
changing gold identity counts/positive labels leaves reference selection unchanged.

```bash
.venv-proposed/bin/python -m pytest -q
```

Optional native CLIP tests run when that library is installed. FAFA's official
imports/preprocessor can be checked in its own environment before downloading
the full runtime assets. Automated CPU tests do not establish pretrained model
quality, full-gallery GPU throughput or VRAM use.

Caches are separate by method and fingerprint checkpoint hashes, ordered image
paths/sizes/mtimes, preprocessing/settings and adapter version. Query caches also
include the selected boxes and text. Completed arrays are published with atomic
rename; interrupted `.tmp` arrays are not reused. Replacing inputs creates a new
cache directory. Existing CPR caches have no matching RCR signature and are not
silently reused. Gallery feature precision is recorded; FAFA defaults to float16
storage and float32 scoring. Cache directories can be removed to reclaim space.

For repeatable runs, retain `run.json`, the repository/source commits, dataset
snapshot and a package snapshot from each solution's Python:
`python -m pip freeze --exclude-editable > requirements-tested.txt`.
Treat that as a platform-specific installed-package snapshot, not a universal
cross-platform lockfile. Environments and downloaded artifacts stay outside Git.
