# Reproduction workflows

`tools/` contains individual Python commands. `scripts/` contains Bash recipes
that run those commands in order and stop on the first failure. Both use the
same functions under `src/rcr/`; no model logic or metrics are duplicated.

Scripts locate the repository root themselves. Direct `tools/*.py` commands
must run from that root after an editable package install. Use the tools when
you need a custom YAML, `--set` overrides, a smoke run or an individual stage.

## Environments

```bash
PYTHON=python3.11 bash scripts/setup.bash clip
```

Replace `clip` with `proposed`, `fafa`, `dataset`, `dataset-clip`, `evaluation` or
`dev`. This creates `.venv-<name>` and installs the matching requirements file.
The host needs Python 3.11/3.12 and pip 22.3+ for `pip --python`. The bootstrap
pins preserve upstream CLIP/FAFA installation support. Dependencies are declared
once in `pyproject.toml`; each requirements file chooses the corresponding extra.
Install only the environments you use. A base install with `requirements.txt`
supports deterministic annotation and JSON reports without Torch.

Proposed requires a separate FAFA environment only to extract FAFA person
features. Configure `person_encoder.python` in `configs/proposed.yaml`.
Training/retrieval from completed caches need only the proposed environment.

## Dataset

```bash
bash scripts/setup.bash dataset
bash scripts/data/prepare_review.bash
.venv-dataset/bin/python -m labelstudio.review.app
# After saving reviews and stopping the review UI:
bash scripts/data/prepare_positives.bash
.venv-dataset/bin/python -m labelstudio.positives.app
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

Each workflow defaults to its `.venv-<method>/bin/python`. Set `PYTHON` to reuse
another matching environment, for example:

```bash
PYTHON=/path/to/clip-env/bin/python bash scripts/methods/clip.bash test
```

Test never trains or selects parameters. Run validation first. Proposed's val
recipe uses `configs/proposed.yaml` and saves the exact run config; it includes
training and requires a new output directory if `last.pt` exists. Its test recipe
loads that saved config and selected checkpoint. For cache reuse without
retraining, another seed/config, or fixed coarse/top-500/full-gallery controls
with `configs/ablations/shortlist.yaml`, call `tools/run.py` directly. Raw images
and model assets must be available as described in the method guides.

## Reports

Use any base/method environment to export already evaluated JSON:

```bash
.venv-dataset/bin/python tools/report.py --split val \
  --run proposed=runs/proposed-v2/val
.venv-dataset/bin/python tools/report.py --split test \
  --run proposed=runs/proposed-v2/test
```

To evaluate saved `rankings.pt` first, use a method environment or install
`requirements/evaluation.txt`; this needs Torch but no encoder libraries.
