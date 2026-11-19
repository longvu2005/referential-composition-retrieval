# Evaluation and reports

Follow the [run guide](proposed_runs.md) and
[benchmark protocol](benchmark_protocol.md) when producing results.

## Selected validation and test

Training exports the selected validation result. After selection, run test and
export both reports:

```bash
.venv-proposed/bin/python tools/report.py --split val
bash scripts/methods/proposed.bash test
.venv-proposed/bin/python tools/report.py --split test
```

The default report input is `runs/proposed/selected/<split>`. Select another
result directory with `--run proposed=DIRECTORY`.

## Saved-ranking evaluation

Evaluate an existing ranking with its matching configuration:

```bash
.venv-proposed/bin/python tools/run.py evaluate \
  --config runs/proposed/selected.yaml --split val
```

Evaluation checks the saved split and binds regenerated metrics to run metadata.
Reporting reads these saved results.

## Exported tables

Reports are written to `runs/reports/<split>/`.

| Files | Contents |
| --- | --- |
| `overall.csv`, `overall.md`, `overall.tex` | Overall benchmark metrics |
| `by_case.csv`, `by_case.md`, `by_case.tex` | Metrics grouped by query case |

CSV uses fractions; Markdown and LaTeX use percentages. `--formats` and
`--decimals` control presentation. Reports validate saved-run consistency before
replacing the active exports.

Keep the effective configuration, selected checkpoint and run metadata with
reported results. Use validation for model selection and test for final results.
