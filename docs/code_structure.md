# Code responsibilities

The method YAML describes an experiment. Functions own one pipeline stage;
there is no registry, recursive config inheritance or trainer framework.
Run Python tools from the repository root after an editable package install.
Bash workflows locate that root before calling tools. YAML paths are relative
to that working directory.

| Responsibility | File |
| --- | --- |
| Commands and existing-key overrides | `tools/run.py`, `src/rcr/common/config.py` |
| Reproduction workflows | `scripts/data/`, `scripts/methods/` |
| Notebook dependency installation | `scripts/setup.bash`, `requirements/`, `pyproject.toml` |
| Dataset/split protocol | `src/rcr/common/data.py` (unchanged) |
| Primary metrics and result validation | `src/rcr/evaluation/` (unchanged) |
| Frozen DINO extraction and letterbox transforms | `src/rcr/proposed/cache/dino.py` |
| Frozen FAFA extraction in current runtime | `src/rcr/proposed/cache/fafa.py`, `tools/cache_fafa.py` |
| Source cache storage / LRU | `src/rcr/proposed/cache/store.py` |
| CLIP cache, resume, provenance and separate labels | `src/rcr/proposed/cache/clip.py` |
| Stage selection | `src/rcr/proposed/cache/build.py` |
| Frozen CLIP extraction, windows and textual Subject parsing | `src/rcr/proposed/nn/encoders.py` |
| Label-free inputs and train supervision | `src/rcr/proposed/batch.py` |
| Grounding, composition, matching, binding, reasoning | Corresponding `src/rcr/proposed/nn/` files |
| Architecture and gradient boundaries | `src/rcr/proposed/nn/model.py` |
| Shared finite identity score | `src/rcr/proposed/scores.py` |
| Supervised and multi-positive losses | `src/rcr/proposed/losses.py` |
| Uniform train-only candidate sampling | `src/rcr/proposed/sampling.py` |
| Coarse/fine/full ranking and train-only hard mining | `src/rcr/proposed/ranking.py` |
| Warmup/main loop, overall-mAP checkpoint selection | `src/rcr/proposed/train.py` |
| Strict checkpoint load and output metadata | `src/rcr/proposed/retrieve.py` |
| Stage/policy coordination | `src/rcr/proposed/experiments.py` |
| Synthetic numerical and end-to-end checks | `tests/proposed/` |

Backbones exist only during cache building. No trainable head output is persisted
in the raw caches. Gallery identity projections refresh for every retrieval or
mining call. Grounding owns its own projections, so detached memberships cannot
be bypassed by a shared trainable text/image head.

Model input dictionaries contain visual/text tensors only. A separate supervision
dictionary carries train identity labels, categorical grounding labels,
correspondence/null labels and candidate positive/valid masks. The unchanged
annotation and baseline packages remain independent of the new architecture.

## Package boundaries

Tools parse arguments and call stage functions. Bash scripts sequence tools and
contain no model or annotation logic. Models/losses never import tools.
`common/` provides configuration, data, I/O, vision and device utilities; data
loading uses the shared annotation parser in `dataset/text.py`.
`evaluation/` evaluates complete split-gallery rankings without model weights.
Proposed and native FAFA use the same Python/runtime. The FAFA cache worker
uses the caller's `sys.executable`; cached training/retrieval never launch it.
`baselines/fafa.py` restores three legacy Transformers utility import aliases
and the Q-former prediction-bias tie before constructing the native model.
The pinned source stays unchanged. The base package supports annotation and JSON reports without
Torch; method extras, optional candidate ordering and saved-ranking evaluation
have separate requirements files.

Keep source annotations, reviewed decisions and the final dataset snapshot.
Runtime caches, checkpoints, rankings and reports are generated and ignored by
Git. Package `__init__.py` files should stay; builders create output directories
when needed. DINO owns scene patches, detector boxes and aligned labels; FAFA
owns a person index tied to the same source/image order. CLIP adds frozen semantic
features and separate supervision sidecars.
