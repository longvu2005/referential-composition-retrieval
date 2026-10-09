# Reproduction workflows

`tools/` contains individual Python commands. `scripts/` contains Bash recipes
that run those commands in order and stop on the first failure. Both use the
same functions under `src/rcr/`; no model logic or metrics are duplicated.

Scripts locate the repository root themselves. Direct `tools/*.py` commands
must run from that root after an editable package install. Use the tools when
you need a custom YAML, `--set` overrides, a smoke run or an individual stage.

## Notebook setup

```bash
bash scripts/setup.bash proposed
```

Replace `proposed` with the selected profile: `clip`, `fafa`, `dataset`,
`dataset-clip`, `evaluation` or `dev`. The script installs into the current
Python; it creates no venvs. One notebook installs and runs one method.
Dependencies are declared in `pyproject.toml`; requirements files select extras.
Proposed includes FAFA cache extraction in the same runtime, with Torch 2.6.0,
torchvision 0.21.0 and Transformers 4.57.6 shared by FAFA, CLIP and DINOv3.
Torch 2.6 is required by Transformers for the pinned CLIP revision's `.bin`
checkpoint. FAFA's legacy utility imports are bridged in
`baselines/fafa.py` without editing the authors' source or model computations.
The loader also restores the native Q-former prediction-bias tie before
`from_pretrained` and vocabulary resizing for BLIP-2's extra `[DEC]` token.

In Colab/Kaggle, run this before shell setup/workflows:

```python
import os
import sys

os.environ["PYTHON"] = sys.executable
```

Run `!bash scripts/setup.bash proposed` from the repository root. Install before
importing models; restart the kernel if packages were already imported. Bootstrap
pins preserve OpenAI CLIP's `pkg_resources` build support. `requirements.txt`
provides deterministic annotation and JSON reports without model dependencies.
The FAFA cache subprocess uses `sys.executable` from the caller; it runs in the
same runtime and ends before CLIP extraction to release GPU/library state.

Setup passes `--no-cache-dir` to pip so large wheels are not retained alongside
installed packages. This does not remove an older pip cache; use
`python -m pip cache info` and `python -m pip cache purge` if notebook disk space
is tight. Proposed checks cache/checkpoint/ranking writes with a 1 GiB reserve
and estimates remaining CLIP cache space before downloading its weights.
See [disk-space recovery](../docs/proposed_runs.md#notebook-disk-space).
Once compatible frozen FAFA features are complete, proposed preparation skips
native FAFA downloads. The default config releases its recorded downloadable
base assets before the CLIP stage; set `cache.release_fafa_assets=false` to retain
them. Tuned checkpoints and feature caches are retained.

When combining stages in one notebook cell, stop the outer shell on errors too:

```bash
%%bash
set -euo pipefail
bash scripts/setup.bash proposed
bash scripts/methods/proposed.bash val
python tools/run.py ablate --config configs/ablations/shortlist.yaml --splits val
```

Pip may report conflicts with unused packages preinstalled by Colab/Kaggle.
Those warnings are distinct from a failed model/cache stage; do not continue
to retrieval or ablation when setup or training actually exits with an error.

## Dataset

```bash
bash scripts/setup.bash dataset
bash scripts/data/prepare_review.bash
python -m labelstudio.review.app
# After saving reviews and stopping the review UI:
bash scripts/data/prepare_positives.bash
python -m labelstudio.positives.app
# After saving positives and stopping the positive UI:
bash scripts/data/finalize.bash --version 0.2.0
```

Review preparation sequences `select_samples.py` and `prepare_review.py`.
Finalization sequences positive normalization and `build_final.py`.
Manual review and positive decisions stay between these workflows. New tasks
are never automatically completed. See [dataset contracts](../dataset/README.md).

## Methods

| Workflow | `val` (default) | `test` |
| --- | --- | --- |
| `scripts/methods/clip.bash` | Prepare assets, run all four variants, select each fusion on val | Reuse frozen fusion weights |
| `scripts/methods/fafa.bash` | Prepare assets, run FAFA, lock adapter protocol | Reuse the protocol lock |
| `scripts/methods/proposed.bash` | Prepare/build DINO, FAFA and CLIP caches, train/select best.pt, evaluate val | Use `runs/proposed-v2/run_config.yaml` and selected best.pt |

```bash
bash scripts/methods/clip.bash val
bash scripts/methods/fafa.bash val
bash scripts/methods/proposed.bash val
```

Each workflow uses the current `python` from PATH, or `PYTHON` when set.
For notebook kernels, use the `sys.executable` binding above. Run validation
first, then the same method's test workflow in that notebook.

Test never trains or selects parameters. Run validation first. Proposed's val
recipe uses `configs/proposed.yaml` and saves the exact run config; it includes
training and requires a new output directory if `last.pt` exists. Its test recipe
loads that saved config and selected checkpoint. For cache reuse without
retraining, another seed/config, or fixed coarse/top-500/full-gallery controls
with `configs/ablations/shortlist.yaml`, call `tools/run.py` directly. Raw images
and model assets must be available as described in the method guides.

## Reports

Use the notebook Python to export already evaluated JSON:

```bash
python tools/report.py --split val \
  --run proposed=runs/proposed-v2/val
python tools/report.py --split test \
  --run proposed=runs/proposed-v2/test
```

To evaluate saved `rankings.pt` first, use a method profile or install
`requirements/evaluation.txt`; this needs Torch but no encoder libraries.
