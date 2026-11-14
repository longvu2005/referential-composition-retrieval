# Reproduction workflows

`tools/` contains individual Python commands. `scripts/` contains Bash recipes
that sequence those commands and stop on the first failure. Both call the same
functions under `src/rcr/`.

Scripts locate the repository root themselves. Run direct `tools/*.py` commands
from that root after an editable package install. Use individual tools for a
custom YAML, `--set` overrides or a single stage.

## Environments

Use Python 3.11 or 3.12:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash person-encoder
```

Setup creates `.venv-<name>` and installs the matching requirements file. Other
supported environments are `dataset`, `dataset-clip`, `evaluation` and `dev`.
Dependencies are declared in `pyproject.toml`; requirements files select the
corresponding extras. The host needs pip 22.3+ for `pip --python`. Bootstrap pins
support the upstream packages used by the frozen person encoder.

The proposed runtime requires PyTorch >=2.6. Frozen FAFA extraction uses PyTorch
2.5.1 and a separate Transformers version in `.venv-person-encoder`. Configure
`person_encoder.python` and `person_encoder.fafa_config` in
`configs/proposed.yaml`; encoder assets are defined in
`configs/person_encoder.yaml`. Completed caches support training and retrieval
using only the proposed environment.

A base install with `requirements.txt` supports annotation and JSON reports
without Torch. `requirements/dataset-clip.txt` adds optional CLIP ordering for
annotation candidates. `requirements/evaluation.txt` adds Torch to evaluate
saved rankings without loading encoders.

## Dataset

```bash
bash scripts/setup.bash dataset
bash scripts/data/prepare_review.bash
.venv-dataset/bin/python -m labelstudio.review.app
# Save reviews and stop the review UI before preparing positives.
bash scripts/data/prepare_positives.bash
.venv-dataset/bin/python -m labelstudio.positives.app
# Save positives and stop the positive UI before finalization.
bash scripts/data/finalize.bash --version 0.2.0
```

Review preparation sequences `select_samples.py` and `prepare_review.py`.
Finalization sequences positive normalization and `build_final.py`.
Manual review and positive decisions stay between these workflows. See
[dataset contracts](../dataset/README.md).

## Proposed method

```bash
bash scripts/methods/proposed.bash val
bash scripts/methods/proposed.bash test
```

Validation prepares assets, builds caches, trains, retrieves validation rankings
and selects the validation artifact through `configs/calibration.yaml`. Test
uses `runs/proposed-calibration/selected.yaml` and its frozen selection.

The recipe defaults to `.venv-proposed/bin/python`. Set `PYTHON` to use another
matching environment:

```bash
PYTHON=/path/to/proposed-env/bin/python bash scripts/methods/proposed.bash test
```

The validation recipe includes a fresh training run. For cache reuse with an
existing checkpoint, another seed or a custom config, call `tools/run.py`
directly. `run` retrieves and evaluates one split; asset preparation, cache
construction and training have separate commands. Training uses pooled FAFA
image features and reads native query/target tokens for context after warmup.
The query cache must cover train before training starts.

Raw images and pretrained assets must be available as described in the
[experiment protocol](../docs/proposed_runs.md). Test uses the checkpoint,
matching settings, pooling and residual weight selected on validation.

## Reports

Export evaluated JSON from any environment with the base package installed:

```bash
.venv-dataset/bin/python tools/report.py --split val \
  --run proposed=runs/proposed-calibration/selected/val
.venv-dataset/bin/python tools/report.py --split test \
  --run proposed=runs/proposed-calibration/selected/test
```

See the [reporting guide](../docs/experiments.md) for table formats and saved-run
validation.
