# Workflows and commands

`tools/` provides individual commands; `scripts/` sequences them into workflows.
Both use the implementations in `src/rcr/`. Run direct Python commands from the
repository root after installing the package.

## Environments

Use Python 3.11 or 3.12:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash person-encoder
```

Setup creates `.venv-<name>` and installs its requirements. Other environments
include `dataset`, `dataset-clip`, `evaluation` and `dev`. Rerun setup to complete
an interrupted installation.

The proposed and person-encoder environments use separate PyTorch and
Transformers versions. Configure `person_encoder.python` for extraction.
Compatible completed caches support training and retrieval in the proposed
environment. Set `PYTHON` to select another interpreter for a workflow.

## Dataset workflows

| Command | Stage |
| --- | --- |
| `bash scripts/data/prepare_review.bash` | Select records and prepare text review |
| `bash scripts/data/prepare_positives.bash` | Prepare positive selection tasks |
| `bash scripts/data/finalize.bash --version VERSION` | Normalize decisions and build final data |

Complete the manual annotation stage between preparation steps. See the
[dataset guide](../dataset/README.md) and [annotation UIs](../labelstudio/README.md).

## Proposed method

```bash
bash scripts/methods/proposed.bash val
bash scripts/methods/proposed.bash test
```

Validation prepares assets, builds caches and trains with checkpoint selection.
Test reads the selected configuration under `output.dir`. Use the Python CLI
for individual stages or custom configurations:

| `tools/run.py` command | Purpose |
| --- | --- |
| `prepare` | Prepare frozen encoder assets |
| `build-cache` | Build or resume feature extraction |
| `prepare-supervision` | Prepare train annotation alignment |
| `train` | Train and select a model; `--resume` continues a run |
| `retrieve` | Generate rankings |
| `evaluate` | Evaluate saved rankings |
| `run` | Retrieve and evaluate a split |
| `calibrate` | Select coefficients from saved validation scores |

Each command takes `--config`; `--set key=value` overrides existing keys.
See the [run guide](../docs/proposed_runs.md) for examples and outputs.

## Reports

```bash
.venv-proposed/bin/python tools/report.py --split val
.venv-proposed/bin/python tools/report.py --split test
```

Reports export saved metrics to CSV, Markdown and LaTeX. See the
[reporting guide](../docs/experiments.md) for input paths and options.
