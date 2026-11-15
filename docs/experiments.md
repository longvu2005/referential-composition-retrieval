# Proposed experiments and paper reports

Use a clean committed checkout and the environments described in the
[script reference](../scripts/README.md). Follow the
[benchmark protocol](benchmark_protocol.md) and
[proposed experiment protocol](proposed_runs.md). Original images, pretrained
assets and historical run outputs are not committed.

## Validation and selection

```bash
bash scripts/methods/proposed.bash val
python tools/report.py --split val
```

The validation recipe prepares assets, builds caches and trains. Each scheduled
validation retrieves once and searches the 729 triples under
`evaluation.calibration`: `coverage_weights`, `context_weights` and
`fine_coarse_weights`. Search runs on CPU with pooling fixed. Checkpoints
compete using the calibrated score. Training saves the winner's `selected.yaml`
and complete validation result under `runs/proposed/selected/val/`.

Optional offline comparison reuses the same calibration implementation:

```bash
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
```

Use a fresh `output.dir` for each training experiment. The optional calibration
YAML lists exported validation artifacts; its default source is the training
winner's `selected/val/rankings.pt`. Check fingerprints,
query/gallery/case counts, encoder text statistics and Git state before
interpreting results. Validation selects parameters and checkpoints; test
provides the final evaluation.

Candidate summaries report the three coefficients, complete-gallery Full-mAP,
macro-case Full-mAP, checkpoint score and coarse candidate recall. Calibration
caches the fixed tail and reuses the official AP summation; the winner receives
the full evaluator, including identity, recall and branch diagnostics. Only the
winning triple's full rankings are retained.

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
python tools/report.py --split val
python tools/report.py --split val --formats csv tex --decimals 3 \
  --run proposed=runs/proposed/selected/val
```

The default input is `runs/proposed/selected/<split>`. `--runs-root` changes that root;
`--run proposed=DIRECTORY` selects a specific split result folder. Pass the
calibrated result directory when reporting the selected model. Reporting
validates all inputs before writing tables and overwrites active exports.

## Saved-ranking evaluation

To evaluate saved rankings without encoder inference, use the proposed or
evaluation environment with the configuration that produced them:

```bash
.venv-proposed/bin/python tools/run.py evaluate --config runs/proposed/selected.yaml --split val
python tools/report.py --split val
```

Use `--set output.dir=...` for another result root. Evaluation checks the saved
split fingerprint against the current annotations and binds the regenerated
metrics to `run.json`. Complete query coverage, the original split fingerprint
and consistent per-query metrics are required for reports.

## Frozen test

After training has selected a calibrated checkpoint on the current validation export:

```bash
bash scripts/methods/proposed.bash test
python tools/report.py --split test
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
