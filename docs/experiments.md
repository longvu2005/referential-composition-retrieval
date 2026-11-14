# Proposed experiments and paper reports

Use a clean committed checkout and the environments described in the
[script reference](../scripts/README.md). Follow the
[benchmark protocol](benchmark_protocol.md) and
[proposed experiment protocol](proposed_runs.md). Original images, pretrained
assets and historical run outputs are not committed.

## Validation and selection

```bash
bash scripts/methods/proposed.bash val
python tools/report.py --split val --run proposed=runs/proposed-calibration/selected/val
```

The validation recipe prepares assets, builds caches, trains, retrieves and
calibrates. Calibration compares saved validation artifacts while retaining the
trained residual weight and pooling. For an existing checkpoint, retrieve from
its saved configuration and calibrate separately:

```bash
.venv-proposed/bin/python tools/run.py run --config runs/proposed/config.yaml --split val
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
```

Use a fresh `output.dir` for each training experiment. List its validation
`rankings.pt` under `configs/calibration.yaml: rankings`. Check fingerprints,
query/gallery/case counts, encoder text statistics and Git state before
interpreting results. Validation selects parameters and checkpoints; test
provides the final evaluation.

## Tables

Reporting reads the proposed method's saved `metrics.json` and `run.json`.
It writes:

```text
runs/reports/<val|test>/
  overall.csv  overall.md  overall.tex
  by_case.csv  by_case.md  by_case.tex
```

Overall has eight metrics. By-case has four metrics for each case, with grouped
LaTeX headers. CSV retains fractions in [0,1]; Markdown and LaTeX use percentages
with two decimals by default. An absent case is blank/—/--. Diagnostic rankings
and shortlist metrics stay in the evaluation output.

```bash
python tools/report.py --split val --run proposed=runs/proposed-calibration/selected/val
python tools/report.py --split val --formats csv tex --decimals 3 \
  --run proposed=runs/proposed-calibration/selected/val
```

The default input is `runs/proposed/<split>`. `--runs-root` changes that root;
`--run proposed=DIRECTORY` selects a specific split result folder. Pass the
calibrated result directory when reporting the selected model. Reporting
validates all inputs before writing tables and overwrites active exports.

## Saved-ranking evaluation

To evaluate saved rankings without encoder inference, use the proposed or
evaluation environment with the configuration that produced them:

```bash
.venv-proposed/bin/python tools/run.py evaluate --config runs/proposed/config.yaml --split val
python tools/report.py --split val
```

Use `--set output.dir=...` for another result root. Evaluation checks the saved
split fingerprint against the current annotations and binds the regenerated
metrics to `run.json`. Complete query coverage, the original split fingerprint
and consistent per-query metrics are required for reports.

## Frozen test

Run validation and calibration on the current dataset export before test:

```bash
bash scripts/methods/proposed.bash test
python tools/report.py --split test --run proposed=runs/proposed-calibration/selected/test
```

Test verifies the selected checkpoint, validation fingerprint and retrieval
settings. Report sample and case counts alongside the tables. CPU tests use
small local fixtures; benchmark experiments require the raw images, pretrained
assets and compatible feature caches:

```bash
python -m pytest -q
python -m ruff check src tools tests labelstudio
python -m ruff format --check src tools tests labelstudio
```
