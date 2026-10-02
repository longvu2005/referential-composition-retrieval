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
`bin/python` with `Scripts/python.exe`. On Kaggle, call the chosen environment's
Python through `subprocess.run([...], check=True)`; shell activation does not
change the notebook kernel. Environments can be rebuilt each session while
images, checkpoints, feature caches and results are restored separately.

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

Inference checks the source commit, real tracked edits and asset inventory before
expensive detection. Official imports disable bytecode writes. The FAFA model
loads from prepared assets with networking blocked; CLIP and the detector also
load explicit local checkpoints. Use `--force` in preparation only when replacing
artifacts is intended. Existing local checkpoints and valid caches can be reused.

## Retrieval and evaluation

CLIP modes share one gallery cache. Choose the same mode/split for retrieve and
evaluate; output placeholders resolve identically in both commands.

```bash
.venv-clip/bin/python tools/methods/retrieve_baseline.py \
  --config configs/methods/baselines/clip.yaml --mode image --split val
.venv-proposed/bin/python tools/methods/evaluate.py \
  --config configs/methods/baselines/clip.yaml --mode image --split val
```

Repeat with `--mode text` and `--mode late_fusion`. `early_fusion` is an additional
ported variant, not a separate implementation or environment.

```bash
.venv-fafa/bin/python tools/methods/retrieve_baseline.py \
  --config configs/methods/baselines/fafa.yaml --split val
.venv-proposed/bin/python tools/methods/evaluate.py \
  --config configs/methods/baselines/fafa.yaml --split val

.venv-proposed/bin/python tools/methods/evaluate.py \
  --config configs/methods/proposed/evaluate.yaml
```

For a smoke run, add the same `--max-queries 4` to retrieve and evaluate. This
selects a query prefix only; the split gallery is still complete. `run.json`
explicitly marks query subsets. A later full run replaces that output with the
full query results and removes previous metrics; run evaluation again for the
new rankings. Baselines have no benchmark training/checkpoint selection
at this stage. Tune heuristic settings on validation only, then freeze them.
The checked-in test split has 14 queries and is a pipeline check, not a mature
publication test set.

Outputs per method/mode/split:

```text
scores.npy      float32 [num_queries, num_split_gallery], higher is better
rankings.pt     sample_ids, gallery_ids, int32 rankings [Q, G-1]
run.json        resolved config, versions, checkpoint hash, cache IDs, diagnostics
metrics.json    official evaluator output (after the separate evaluate command)
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
| image | Normalized whole-scene image embedding | Image dot gallery image |
| text | Normalized `final_instruction` embedding | Text dot gallery image |
| early_fusion | Normalized weighted sum of image/text embeddings | Fused query dot gallery image |
| late_fusion | Image/text branch scores | Per-query population z-score of each branch, weighted sum |

CLIP uses the pinned OpenAI implementation and `ViT-L/14`. Weights start at
0.5/0.5, matching the old configs. Alternative `final_change` text is an explicit
config variant. Truncated text counts are recorded; no LLM rewriting occurs.

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
