# Editing the research code

The method YAML describes an experiment. Functions implement its stages directly;
there is no registry, recursive config inheritance or trainer framework.
Run scripts from the repository root after installing the package in editable mode.
Relative YAML paths are resolved from that working directory.

## Where to change behavior

| Change | Start here |
| --- | --- |
| Experiment commands and dispatch | `scripts/run.py` |
| YAML loading and dotted overrides | `src/rcr/common/config.py` |
| Device selection | `src/rcr/common/runtime.py` |
| Final data loading and split gallery | `src/rcr/common/data.py` |
| Model architecture | `src/rcr/proposed/nn/model.py`, then the component in `nn/` |
| Loss terms and weights | `src/rcr/proposed/losses.py`, `configs/proposed.yaml` |
| Identity, state and binding scores | `src/rcr/proposed/scores.py` |
| Training and checkpoint selection | `src/rcr/proposed/train.py` |
| Candidate sampling and hard negatives | `src/rcr/proposed/sampling.py`, `ranking.py` |
| Batch assembly and prefetch | `src/rcr/proposed/batch.py` |
| Shortlisting, normalization and reranking | `src/rcr/proposed/ranking.py` |
| Checkpoint loading and result metadata | `src/rcr/proposed/retrieve.py` |
| Stage coordination, ablations and validation calibration | `src/rcr/proposed/experiments.py` |
| DINO extraction | `src/rcr/proposed/cache/dino.py` |
| FAFA extraction in its own environment | `src/rcr/proposed/cache/fafa.py`, `scripts/cache_fafa.py` |
| Cache stage selection and provenance checks | `src/rcr/proposed/cache/build.py` |
| Cache loading used by both coarse and fine | `src/rcr/proposed/cache/store.py` |
| CLIP/FAFA baselines and validation tuning | `src/rcr/baselines/` |
| Saved/in-memory evaluation orchestration | `src/rcr/evaluation/runner.py` |
| Official protocol and metric definitions | `src/rcr/evaluation/evaluate.py`, `metrics.py` |
| Dataset construction and annotation | `src/rcr/dataset/`, `scripts/data/`, `labelstudio/` |

## Boundaries

- Scripts parse arguments and call stage functions. Models/losses never import scripts.
- `common/` provides configuration, data, I/O, vision and device utilities.
  It does not import evaluation or either retrieval method. Data loading uses the
  shared annotation parser in `dataset/rewrite.py` to interpret Subject descriptions.
- `evaluation/` evaluates complete split-gallery rankings from any method. It
  does not load model weights. Both proposed and baseline workflows call it.
- Proposed inference constructs its model from the saved checkpoint config.
  Inference ablations change retrieval settings; architecture changes require
  retraining. `best.pt` is selected by the existing average of overall and
  macro-case validation Full-mAP. Calibration selects its coefficients by
  validation Full-mAP; test never selects them.
- Native FAFA dependencies stay in their own environment. Training and retrieval
  consume its frozen features and do not launch the FAFA worker.
- Tests follow the owning modules. CPU tests use small local fake backbones;
  optional native CLIP/Transformers and CUDA checks may skip without those extras.

## Persistent inputs versus generated state

Keep source annotations, reviewed decisions and the final dataset snapshot: they
are research provenance, not disposable files. Runtime caches, checkpoints,
rankings, audit reports and partial exports are generated and ignored by Git.
`__init__.py` files define Python packages and should not be removed as empty junk.
Git does not version empty directories. Builders create their output directories
when needed, so a patch need not add placeholder folders for future experiments.

## Cache contract

DINO owns scene patches, person boxes, GT alignment labels and DINO crop features.
FAFA owns a separate, complete person index tied to that DINO source ID and image
order. Its per-image shards are used only while extracting/resuming. Once the
FAFA index is published, both coarse projection and fine batches read persons
from that index and obtain scenes/boxes/labels from the DINO source.

Older completed FAFA caches with leftover shards also work: the complete index
is authoritative. The patch does not convert or rewrite existing DINO caches,
and it preserves model state-dict keys, loss formulas and metric definitions.
