# Run guide

Run commands from the repository root using the environments described in the
[script reference](../scripts/README.md#environments).

## Configure paths

Edit [configs/proposed.yaml](../configs/proposed.yaml) for the current machine
and experiment:

| Setting | Purpose |
| --- | --- |
| `data.final_dir`, `data.image_root` | Benchmark annotations and raw images |
| `data.*_cache` | Detection, scene and person feature directories |
| `person_encoder` | Extraction environment and encoder configuration |
| `cache.supervision_cache` | Separate train annotation alignment |
| `output.dir` | Run artifacts |

Model, loss, sampling, retrieval and runtime settings are defined in the same
file. `--set key=value` overrides existing configuration keys.
See [GPU performance](gpu_performance.md) for chunking, gallery/text cache budgets
and a CUDA profiling command.

## Prepare assets and caches

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml
```

Cache construction reuses compatible outputs and resumes incomplete stages.
See the [cache guide](detection_cache.md) for independent builds. Training needs
query features for train and validation; retrieval needs the requested split.

## Train and select a checkpoint

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml
```

Use a fresh `output.dir` for each experiment. For example:

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set train.seed=1 output.dir=runs/proposed-seed1
```

Training performs scheduled validation, calibrates the configured scoring
coefficients and saves the best selection. The selection criterion is defined
in the [benchmark protocol](benchmark_protocol.md).
W&B tracking is configured by `wandb`; use `wandb.mode=offline` or
`wandb.enabled=false` when online tracking is unavailable.

## Resume training

Continue the same run using its saved effective configuration:

```bash
.venv-proposed/bin/python tools/run.py train --config runs/proposed/config.yaml --resume
```

An explicit path may follow `--resume`. Resume uses the full training state in
`last.pt`, including optimizer, random generators and sampling progress. Keep
the original run directory, numerical settings, total epoch count, source,
runtime, data and caches compatible with the saved state. `best.pt` is the
inference checkpoint.

## Seeds and reproducibility

`train.seed` controls random initialization and sampling.
`runtime.deterministic` requests reproducible kernels. Record the source
revision, effective configuration, runtime and data/cache identities with each
experiment; hardware and software changes can affect numerical results.

## Inspect validation and optional calibration

Training writes the selected validation result automatically. Export it with:

```bash
.venv-proposed/bin/python tools/report.py --split val
```

Optional offline calibration searches saved validation scores:

```bash
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
```

Configure validation artifacts in
[configs/calibration.yaml](../configs/calibration.yaml). Calibration reuses saved
scores on CPU and exports a selected configuration for test.

## Run test

```bash
.venv-proposed/bin/python tools/run.py run \
  --config runs/proposed/selected.yaml --split test
```

Test uses the validation-selected checkpoint and scoring settings. The runner
checks checkpoint, dataset and cache compatibility before retrieval.

## Reading outputs

Paths below are relative to the configured `output.dir`.

| Artifact | Contents |
| --- | --- |
| `config.yaml`, `runtime.json` | Effective settings and runtime metadata |
| `history.jsonl` | Training and validation history |
| `last.pt` | Full training state for resume |
| `best.pt`, `selected.yaml`, `selection.json` | Selected model and scoring settings |
| `evaluation/` | Scheduled validation summaries |
| `selected/val/`, `selected/test/` | Rankings, run metadata and evaluated metrics |

See the [reporting guide](experiments.md) for evaluation and table export.

## Score artifacts and CPU replay

`rankings.pt` contains rankings, shortlist evidence and provenance. Compatible
artifacts can be replayed without model weights or feature caches:

```bash
.venv-proposed/bin/python -m rcr.proposed.replay \
  runs/proposed/selected/val/rankings.pt --output-dir runs/replayed/val
```

Replay can vary supported score coefficients on the saved shortlist. Changes
to the model, matching, query inputs or shortlist require fresh retrieval.
Use validation to select coefficients under the benchmark protocol.
