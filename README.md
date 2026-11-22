# Referential Composition Retrieval

Research code and benchmark for **Referential Composition Retrieval (RCR)**.
A query combines a reference image, Subject descriptions and a requested change.
The method retrieves images that preserve the referenced identities and satisfy
that change.

The proposed RCR model injects aligned target evidence into individual person
tokens and reasons jointly with ordered Subject-pair tokens. One CLS readout
provides the context score; native FAFA binding and identity coverage complete
the fine score. See the [method and formulas](docs/proposed_method.md).

```text
dataset -> feature caches -> training -> retrieval -> evaluation
```

## Setup

Use Python 3.11 or 3.12. Install separate environments for the proposed method
and frozen person encoder:

```bash
PYTHON=python3.11 bash scripts/setup.bash proposed
PYTHON=python3.11 bash scripts/setup.bash person-encoder
```

Prepare the data using the [dataset guide](dataset/README.md). Set local data,
cache and output paths in [configs/proposed.yaml](configs/proposed.yaml).
Frozen encoder assets are configured in
[configs/person_encoder.yaml](configs/person_encoder.yaml).
Raw images and pretrained assets are stored separately from the repository.

## Run

The validation workflow prepares assets, builds or reuses caches, trains the
model and selects a checkpoint. Run test after validation selection completes:

```bash
bash scripts/methods/proposed.bash val
bash scripts/methods/proposed.bash test
```

Continue an interrupted training run with its saved configuration:

```bash
.venv-proposed/bin/python tools/run.py train --config runs/proposed/config.yaml --resume
```

Outputs are written under `output.dir`. `best.pt` and `selected.yaml` describe
the validation-selected model; `last.pt` supports training resume.
The supplied RCR configuration uses hinge rejection and retains the experiment
name `proposed` and output directory `runs/proposed`.
The `fafa-aligned-pair-cls-v5` architecture learns separate binding weights
for query grounding and target reasoning. Train it in a fresh run directory.
Retrieval also supports v4 checkpoints trained with shared binding weights.
See the [run guide](docs/proposed_runs.md) for stages, outputs and the matched
softplus ablation.

## Documentation

| Guide | Contents |
| --- | --- |
| [Method](docs/proposed_method.md) | Model overview and training objectives |
| [Run guide](docs/proposed_runs.md) | Configuration, training, resume and retrieval |
| [GPU performance](docs/gpu_performance.md) | CPU/GPU transfers, synchronization, cache budgets and profiling |
| [Feature contracts](docs/feature_contracts.md) | Encoder, cache and checkpoint compatibility |
| [Cache guide](docs/detection_cache.md) | Cache preparation and reuse |
| [Benchmark protocol](docs/benchmark_protocol.md) | Galleries, positives, metrics and split usage |
| [Reports](docs/experiments.md) | Evaluation and result export |
| [Dataset](dataset/README.md) | Data layout and construction |
| [Annotation UIs](labelstudio/README.md) | Text review and positive selection |
| [Annotation audit](docs/full_positive_audit.md) | Integrity checks and review workflow |
| [Scripts](scripts/README.md) | Environments and command reference |

## Development

Core code lives in `src/rcr/`; command entry points are in `tools/`, workflow
scripts in `scripts/`, and tests in `tests/`.

```bash
bash scripts/setup.bash dev
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/python -m ruff check src tools tests labelstudio
.venv-dev/bin/python -m ruff format --check src tools tests labelstudio
```
