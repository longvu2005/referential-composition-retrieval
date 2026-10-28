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

The training objective separates the components used by retrieval:

```text
L = L_ground + 0.1 L_id + L_coarse + L_context + L_state
```

| Objective | Supervision |
| --- | --- |
| Grounding | Foreground/background-balanced BCE per Subject, averaged per query |
| Crop identity | Supervised contrastive learning on deduplicated train crops |
| Coarse identity | Same-required-ID images outrank wrong-ID images, using the actual coarse identity scorer |
| Context | Full Positives outrank same-ID wrong-state/relation images with nonzero predicted matching coverage |
| Global state | Full Positives outrank same-ID wrong-state images in global CLIP state space |

Pairwise losses average pairs within each query and then eligible queries, so
queries with more positives do not dominate. Wrong-ID images have unknown context
labels and are excluded from context/state supervision. A context query without
both a supported positive and a supported same-ID negative contributes zero;
`context_active_fraction` logs how often this objective is usable. No GT label
changes predicted membership or matching.

Training supervises `coverage * context` directly and does not include frozen
FDA binding in its loss. It reads pooled FAFA image features, without loading
native FAFA query/target tokens or computing FDA. New runs keep `binding_scale=1`
fixed; its existing checkpoint key is retained and recorded values still load.
This prevents the train loss from being reduced by inflating a frozen branch's
coefficient. Calibrating the context weight remains a validation-only step.

Validation/retrieval retains the same formula and full Top-M normalization:

```text
S_fine = alpha * B_FAFA + context_weight * z_topM(coverage * context)
```

`alpha` is the checkpoint's recorded binding scale, and the initial inference
`context_weight` remains 0.0325. Normalization uses one complete Coarse Top-M per
query, FP32 population standard deviation and epsilon `1e-6`. Training candidate
batches do not define inference normalization groups. Constants contribute zero.
Set `retrieval.fine_score_mode=legacy_raw` to use `alpha * B_FAFA + coverage * context`.
Final fusion remains `z(S_fine) + fine_coarse_weight * z(S_coarse)` on common finite
Coarse support; weight zero returns the fine score directly.

Loss weights and cosine temperatures live under `loss`; inference weights live
under `retrieval`. The new objective needs a fresh training run, using a new
`output.dir`. Existing checkpoints keep their inference behavior. If disabling
`loss.state_weight`, also set `retrieval.coarse_beta=0`.

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
trains, retrieves validation scores once, calibrates four scoring parameters
offline, and writes the selected validation results:

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

Use a new `output.dir` for each training run. Training refuses a directory that
already contains `config.yaml`, `best.pt` or `last.pt`. Retrieval, evaluation,
calibration and reports overwrite their active outputs without archiving them.

The CLI has four options: `--config`, `--set`, `--split` and `--cache-stage`.
It always processes complete query splits. CLIP mode and other experiment parameters belong in YAML or `--set`.

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

- `rankings.pt`: rankings, raw scores, scoring settings and embedded provenance;
- `run.json`: configuration, checkpoint, cache and benchmark provenance;
- `metrics.json`: overall and by-case metrics.

`best.pt` maximizes the mean of overall validation Full-mAP and macro-case
validation Full-mAP. With the chosen checkpoint, the official pipeline is:

```bash
python tools/run.py run --config configs/proposed.yaml --split val
python tools/run.py calibrate --config configs/calibration.yaml
python tools/run.py run --config runs/clip-b32-calibration/selected.yaml --split test
```

Validation retrieval saves `identity_scores`, `state_scores`, `fafa_binding`,
`context` and `coverage` as CPU float32 `[Q,G]` matrices in `gallery_ids` order.
It scores the original Top-M first, then each remaining non-self pair once.
This requires approximately `Q*(G-1)` fine pairs and `20*Q*G` bytes for raw
scores, plus diagnostic rankings. Periodic training validation and test retrieval
continue to fine-score only Top-M.

`rankings.pt` is the sole input artifact for offline calibration: it embeds
query/gallery IDs, query indices, recorded binding scale, fixed matching settings,
score configuration, the original run config and provenance. Both
`score_schema_version` and `schema_version` stay **1**. Raw scores are stored once;
shortlist scores/statistics and coarse/fine/FAFA rankings remain as diagnostics.

`configs/calibration.yaml` contains `rankings`, `output_dir` and grids for
`coarse_beta`, `fine_coarse_weight`, `context_weight` and `top_m`. No top-level
`method` or separate model config is required. Calibration reads this artifact
and validation annotations, without opening checkpoint weights, backbone assets,
images or feature caches. It rebuilds coarse order and Top-M, then uses the same
CPU float32 normalization/fusion helpers as direct retrieval. Coarse sorting,
shortlist statistics and fusion normalization are cached across grid tuples.
`candidate_ks` stays fixed from the evaluation config and is measured on the
complete coarse order, independently of Top-M.

Selection maximizes validation Full-mAP and writes `summary.csv`, `selection.json`,
`selected.yaml` and the best `selected/val/{rankings.pt,run.json,metrics.json}`.
Only the best full ranking is serialized. The checkpoint fingerprint, complete
validation fingerprint, query IDs and gallery order are checked; test retrieval
also verifies the checkpoint bytes and frozen scoring settings. Test labels
never select a checkpoint or hyperparameters.

For a single offline ablation, replay the same validation artifact directly:

```bash
python -m rcr.proposed.replay runs/proposed-clip-b32/val/rankings.pt \
  --output-dir runs/replay/val --coarse-beta 0.2 --top-m 500 \
  --context-weight 0.0288 --fine-coarse-weight 0.0064
```

Replay requires the complete version-1 raw-score artifact; rerun validation
retrieval once if the required raw scores are missing. See
[docs/proposed_runs.md](docs/proposed_runs.md#score-artifacts-and-cpu-replay)
for the exact fields and fixed scoring conventions.

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
