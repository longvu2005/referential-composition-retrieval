# Baseline experiments and paper reports

Use a clean committed checkout and separate environments from [baselines](baselines.md).
Follow the [benchmark protocol](benchmark_protocol.md). Original images, pretrained
assets and historical run outputs are not included in the repository.

## Validation first

```bash
.venv-clip/bin/python scripts/run.py run --config configs/clip.yaml --prepare --splits val
.venv-fafa/bin/python scripts/run.py run --config configs/fafa.yaml --prepare --splits val
# Proposed architecture/training is unchanged; use its prepared checkpoint/cache:
.venv-proposed/bin/python scripts/run.py run --config configs/proposed.yaml --splits val
python scripts/report.py --split val
```

CLIP selects fusion weights on Full-mAP. FAFA evaluates/locks the supplied adapter
configuration; tune manually on val if needed. Use separate output roots for
trials: `--set output.dir=runs/<experiment-name>`. Check hashes, fingerprints,
query/gallery/case counts, truncation/fallback statistics and Git state before
interpreting results. Never select methods/configs using test metrics.

Reporting defaults to all four CLIP variants, FAFA and Proposed. It reads only
saved `metrics.json`/`run.json` and exports:

```text
runs/reports/<val|test>/
  overall.csv  overall.md  overall.tex
  by_case.csv  by_case.md  by_case.tex
```

Overall has 8 metrics. By-case has 4 metrics for each case, with grouped LaTeX
headers. CSV retains raw fractions in [0,1]; Markdown/LaTeX use percentages,
2 decimals by default. An absent case is blank/—/--. Diagnostics and automatic
best-method selection are excluded. All selected runs must validate before
writing any table; previous tables are preserved in `.history/`.

```bash
# Use the experiment name recorded in calibration/selection.json:
python scripts/report.py --split val --run proposed=runs/calibration/CHOSEN_EXPERIMENT/val
# Explicitly request a smaller table or alternative formats:
python scripts/report.py --split val --methods clip_image clip_text early_fusion late_fusion fafa
python scripts/report.py --split val --formats csv tex --decimals 3
```

`--run METHOD=DIRECTORY` takes a split result folder; `--runs-root` changes the
default run root. Missing selected methods are errors. Do not replace a selected
calibrated model with an unselected run.
Calibration keeps val outputs in each experiment folder; its frozen test output
is `runs/calibration/selected/test`, which can be passed with `--run proposed=...`.

## Historical saved results

Legacy metrics need a run/evaluation binding. Re-evaluate saved rankings with
the same root/config/mode/split, without inference:

```bash
.venv-clip/bin/python scripts/run.py evaluate --config configs/clip.yaml --splits val \
  --modes clip_image clip_text early_fusion late_fusion
.venv-fafa/bin/python scripts/run.py evaluate --config configs/fafa.yaml --splits val
.venv-proposed/bin/python scripts/run.py evaluate --config configs/proposed.yaml --splits val
python scripts/report.py --split val
```

Use `--set output.dir=...` for historical roots. Evaluation preserves old metrics
and checks the original split fingerprint against current annotations. Missing
original fingerprints, fixed-weight fusion without selection, partial queries
and unfrozen FAFA test runs cannot enter paper tables. Reporting never invents
results or historical Git/runtime metadata.

## Frozen test

Test currently contains only 14 queries, so it is not a publication-scale final
benchmark. After independent expansion/review, validate and lock against the new
export before test. An explicitly requested current test smoke check uses the
same frozen policy:

```bash
.venv-clip/bin/python scripts/run.py run --config configs/clip.yaml --splits test
.venv-fafa/bin/python scripts/run.py run --config configs/fafa.yaml --splits test
.venv-proposed/bin/python scripts/run.py run --config configs/proposed.yaml --splits test
python scripts/report.py --split test
```

Report sample/case counts alongside such tables. CPU tests use tiny local fixtures;
they are not pretrained validation/test experiments:

```bash
python -m pytest -q
python -m ruff check src scripts tests labelstudio
```
