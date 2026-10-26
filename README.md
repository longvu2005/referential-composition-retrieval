# Referential Composition Retrieval

Research code for **Referential Composition Retrieval (RCR)**. A query contains a
reference image, text that identifies one or more Subjects, and a requested
change. The task is to rank gallery images that preserve the referenced
identities and satisfy that change.

The repository deliberately uses plain PyTorch, YAML and small Python modules.
There is no trainer framework, registry, config inheritance or plugin layer. The
main path is:

```text
final dataset -> frozen feature caches -> train -> retrieve -> evaluate
```

## Method

The proposed method keeps extraction frozen and learns only the RCR modules:

1. Ground each Subject description to people detected in the query image.
2. Project person crops into an identity space for coarse retrieval.
3. Match query people to target people with one global partial assignment and a
   private null option for every query person.
4. Score matched pairs with native FAFA composition and relational scene
   context.
5. Rerank the coarse Top-M shortlist with the fine score.

Scene and text backbones are the frozen pair from `openai/clip-vit-base-patch32`,
pinned to revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The scene cache
keeps 49 patch tokens (7 x 7, width 768) and a separate native 512-D global vector.
Text uses the native 77-position tokenizer and projected EOT vector. Subject
roles are located by token offsets; no new vocabulary entries are added. State
adapters start as identity maps, so their initial cosine is native CLIP cosine;
state supervision then adapts those maps. See [migration notes](docs/clip_b32_migration.md).

The training objective is

```text
L = L_ground + lambda_id L_id + L_retrieval + lambda_state L_state
```

`L_ground` is masked BCE, `L_id` is supervised contrastive loss,
`L_retrieval` is pairwise ranking loss, and `L_state` learns the global requested
change only from images that contain the required identities. See
[docs/proposed_method.md](docs/proposed_method.md) for the complete equations.

Training retains the raw fine score `alpha * B_FAFA + coverage * context`.
Validation/retrieval instead use Context z-score with fixed weight 0.0325:

```text
S_fine = alpha * B_FAFA + 0.0325 * z_topM(coverage * context)
```

`alpha` is the checkpoint's learned binding scale. Normalization uses one complete
Coarse Top-M per query, FP32 population standard deviation and epsilon `1e-6`.
Constants contribute zero; inference batch sizes do not define normalization
groups. Set `retrieval.fine_score_mode=legacy_raw` to restore raw fine scoring.
Final fusion remains `z(S_fine) + fine_coarse_weight * z(S_coarse)` on common finite
Coarse support; weight zero returns the fine score directly.

Inference receives only images, text and public Subject slots. Case labels,
identity labels, head boxes and positive sets are used only for training or
evaluation. `load_rcr_inputs()` enforces this boundary.

## Repository layout

```text
configs/                 experiment configuration
dataset/data/final/      finalized benchmark metadata
src/rcr/common/          data, I/O and runtime helpers
src/rcr/dataset/         deterministic dataset construction
src/rcr/baselines/       CLIP and native FAFA baselines
src/rcr/proposed/cache/  frozen CLIP/FAFA extraction and readers
src/rcr/proposed/nn/     learned RCR modules
src/rcr/evaluation/      metrics and reports
tools/run.py             experiment CLI
scripts/                 reproducible commands
tests/                   small mathematical and pipeline contracts
```

Dependency direction is `tools -> experiment functions -> model/shared
utilities`. Files under `nn/` contain the method; `train.py` owns optimization;
`ranking.py` owns inference.

## Setup

Use Python 3.11 or 3.12. Proposed requires PyTorch 2.6 or newer for the
pinned CLIP `.bin` checkpoint; native FAFA keeps PyTorch 2.5.1. Their Torch and
Transformers requirements are installed in separate environments.

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash fafa
PYTHON=python3.11 bash scripts/setup.bash clip
```

Set the paths in `configs/proposed.yaml`, especially:

```yaml
data:
  final_dir: dataset/data/final
  image_root: dataset/data/raw/images
  detection_cache: cache/grounding-dino-tiny
  scene_cache: cache/proposed-clip-b32
  fafa_image_cache: cache/fafa-person-images
  fafa_query_cache: cache/fafa-person-queries
person_encoder:
  python: .venv-fafa/bin/python
```

## Run the proposed method

The complete validation workflow prepares FAFA assets, reuses or builds caches,
trains, calibrates the two fusion weights on validation, and writes validation
results:

```bash
bash scripts/methods/proposed.bash val
```

Run test only after validation has produced a frozen selected configuration:

```bash
bash scripts/methods/proposed.bash test
```

Every stage can also be run directly:

```bash
python tools/run.py prepare     --config configs/proposed.yaml
python tools/run.py build-cache --config configs/proposed.yaml
python tools/run.py train       --config configs/proposed.yaml
python tools/run.py retrieve    --config configs/proposed.yaml --split val
python tools/run.py evaluate    --config configs/proposed.yaml --split val
python tools/run.py calibrate   --config configs/calibration.yaml
```

`run` retrieves and evaluates one split using prepared assets and an existing
checkpoint. Preparation, cache extraction and training are explicit commands;
the full sequence is in `scripts/methods/`. The split comes from YAML (`val` in
the supplied configs), overridden by `--split`. Repeat the command for test.
CLIP `run` evaluates all four modes; `retrieve`/`evaluate` use YAML `mode`.

Build one stage with `--cache-stage detections`, `scene`, `fafa-image`, or
`fafa-query`. Scene extraction needs neither detection nor FAFA caches. FAFA
extraction needs the detection cache, without loading CLIP scene weights.
Use `--set key=value ...` for explicit typed overrides. For example:

```bash
python tools/run.py train --config configs/proposed.yaml \
  --set train.seed=1 output.dir=runs/proposed-seed1
python tools/run.py run --config runs/proposed-seed1/config.yaml --split val
```

The CLI has five options: `--config`, `--set`, `--split`, `--cache-stage` and
`--save-tuning` (Proposed validation only). It always processes complete query
splits. CLIP mode and other experiment parameters belong in YAML or `--set`.

Training logs step losses, gradient norm, learning rates, epoch means,
validation metrics and the best checkpoint summary to W&B. Authenticate once
with `wandb login`; use `--set wandb.mode=offline` when a machine cannot sync,
or `--set wandb.enabled=false` to disable tracking for a smoke test.

## Caches

| Config key | Contents | Depends on query text |
| --- | --- | --- |
| `data.detection_cache` | shared ordered pixel boxes and original image sizes | no |
| `data.scene_cache` | CLIP scene patches and native global embeddings | no |
| `data.fafa_image_cache` | pooled person features and target tokens | no |
| `data.fafa_query_cache` | text-conditioned query crop features | yes |

The four caches are separate: a text edit rebuilds only FAFA query features, and
a scene-encoder change leaves detection and FAFA features reusable. Both FAFA
paths and the standalone FAFA baseline share the detection cache's crop order.
Pixel boxes are mapped to CLIP letterbox coordinates only when joining features.
Visual caches never store training labels. Train-only identity labels are aligned to
cached boxes in `cache.supervision_cache` before the first epoch.

Cache extraction is resumable. A completed cache is reused; an interrupted cache
keeps a `.building` marker and resumes its missing shards. Completed unified CLIP
scene caches remain readable for scene features. Import their boxes (or boxes
from a matching legacy DINO scene cache) through the Python example in the
[detection cache guide](docs/detection_cache.md#import-existing-scene-boxes-without-detector-inference).
Build FAFA features once in the new independent
directories; old scene-bound FAFA indices are not silently rebound. DINO/BERT
model checkpoints require fresh CLIP training. See
[detection cache migration](docs/detection_cache.md).

## Dataset snapshot

Version `0.1.0` contains 4,311 queries and 37,107 images. Each query searches the
images in its own PIPA split, excluding its reference image.

| Split | INDIVIDUAL | GROUP | DUAL | RELATIONAL | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,472 | 137 | 269 | 155 | 4,033 |
| val | 194 | 12 | 36 | 22 | 264 |
| test | 5 | 2 | 3 | 4 | 14 |

The current test split validates the pipeline; it is not a publication-scale
holdout. Raw images are not committed. Dataset reconstruction and annotation
audits are documented in [dataset/README.md](dataset/README.md).

## Outputs

Training writes `config.yaml`, `runtime.json`, `history.jsonl`, `last.pt`, and the
validation-selected `best.pt`. Checkpoints contain learned weights and frozen-model
provenance, but no optimizer state or duplicate frozen CLIP weights. Retrieval
writes one directory per split with:

- `rankings.pt`: fused, coarse, fine and FAFA diagnostic rankings, plus raw and
  normalized score components aligned with `coarse_topm`;
- `run.json`: configuration, checkpoint, cache and benchmark provenance;
- `metrics.json`: overall and by-case metrics.

`best.pt` maximizes the mean of overall validation Full-mAP and macro-case
validation Full-mAP. `configs/calibration.yaml` then selects `coarse_beta` and
`fine_coarse_weight` on validation. Test labels never select a checkpoint or
weight.

Score schema 1 adds `coarse_topm_scores`, `gated_context`, `fine_raw_scores`,
`fine_scores`, `fused_topm_scores`, `context_mean`, `context_std`, `binding_scale`
and `score_config`. It supports CPU replay on the saved shortlist without a
checkpoint or pretrained caches:

```bash
python -m rcr.proposed.replay runs/clip-b32-calibration/selected/val/rankings.pt \
  --output-dir runs/replay/val --mode context_zscore --context-weight 0.0325 \
  --fine-coarse-weight 0.4
```

See [docs/proposed_runs.md](docs/proposed_runs.md) for the full schema and offline
ablations. The evaluator still accepts old rankings; offline replay requires the
new score components. Model inference requires a freshly trained CLIP checkpoint.
With Top-M-only artifacts, changing `coarse_beta` may change shortlist membership
and requires retrieval again.

For offline tuning that can change the shortlist, export full-gallery raw scores
once with the existing checkpoint and caches (validation only):

```bash
python tools/run.py run --config runs/proposed-clip-b32/config.yaml \
  --split val --save-tuning --set output.dir=runs/proposed-clip-b32-tuning
python -m rcr.proposed.replay runs/proposed-clip-b32-tuning/val/rankings.pt \
  --output-dir runs/replayed/val --coarse-beta 0.2 --top-m 500 \
  --context-weight 0.0288 --fine-coarse-weight 0.0064
```

`--save-tuning` scores every non-self query/gallery pair once and stores raw
identity/state and FAFA/context/coverage scores inside `rankings.pt`. This extra
work is opt-in and is not added to periodic training validation. Later replay
needs only that file and CPU PyTorch; it rebuilds the complete Coarse ranking,
new shortlist, its normalization and the new Coarse tail. No model, image
features or backbone downloads are needed. Old Top-M-only artifacts retain their
existing replay support but cannot recover candidates they never scored.
See [docs/proposed_runs.md](docs/proposed_runs.md#full-gallery-tuning-export)
for fields, cost, provenance and fixed matching settings.

To search weights on the export and freeze a test-ready configuration without
running model inference, use:

```bash
python tools/run.py calibrate --config configs/calibration_saved.yaml
```

This CPU-only path evaluates the complete validation split, writes
`summary.csv`, `selection.json` and `selected.yaml`, and retains only the best
full ranking under `selected/val/`. It checks the export's checkpoint fingerprint
and exact validation text/labels/ordered gallery; no checkpoint file or visual
feature cache is opened during this selection.

## Development

```bash
bash scripts/setup.bash dev
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/python -m ruff check src tools tests labelstudio
.venv-dev/bin/python -m ruff format --check src tools tests labelstudio
```

When changing the method, edit the module that owns the equation, use a new
output directory, and record the commit, config and seed. Keep preprocessing and
evaluation fixed so an ablation changes one research decision at a time.
