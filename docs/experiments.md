# Experiment workflow

Run one method per notebook, using its own config and the shared evaluator.
Install that method profile into the notebook Python before running workflows. Baseline definitions, fusion validation and metrics are unchanged;
see [baselines](baselines.md). Proposed v2 uses the architecture in
[proposed_method.md](proposed_method.md) and commands in
[proposed_runs.md](proposed_runs.md).

```bash
bash scripts/methods/clip.bash val
bash scripts/methods/fafa.bash val
bash scripts/methods/proposed.bash val
python tools/run.py ablate --config configs/ablations/shortlist.yaml --splits val
bash scripts/methods/proposed.bash test
python tools/report.py --split val --run proposed=runs/proposed-v2/val
```

Only overall validation Full-mAP selects the proposed checkpoint. Case metrics
are reporting groups only. Warmup/legacy/oracle results never select a primary
checkpoint. Keep each variant in a separate output directory, especially full
fine versus shortlist approximation. Train outputs are not overwritten if they
already contain `last.pt`.

The repository's evaluator requires a complete gallery permutation excluding
self. Proposed shortlisting reorders its first M items with the exact fine score,
then appends the coarse tail. Full fine uses the same positives and denominator.
Use matching `--max-queries` for retrieve/evaluate environment smoke runs; these
are labelled subsets and should not be reported as full split experiments.

## Reports

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
# Use the selected v2 checkpoint output:
python tools/report.py --split val --run proposed=runs/proposed-v2/val
# Explicitly request a smaller table or alternative formats:
python tools/report.py --split val --methods clip_image clip_text early_fusion late_fusion fafa
python tools/report.py --split val --formats csv tex --decimals 3
```

`--run METHOD=DIRECTORY` takes a split result folder; `--runs-root` changes the
default run root. Missing selected methods are errors. Use the same selected
checkpoint and fixed retrieval policy for validation and test. Proposed test
outputs are in `runs/proposed-v2/test`, which can be passed with `--run proposed=...`.

## Historical saved results

Legacy metrics need a run/evaluation binding. Re-evaluate saved rankings with
the same root/config/mode/split, without inference:

```bash
python tools/run.py evaluate --config configs/clip.yaml --splits val \
  --modes clip_image clip_text early_fusion late_fusion
python tools/run.py evaluate --config configs/fafa.yaml --splits val
# For old proposed rankings, use their saved historical config and output root:
python tools/run.py evaluate --config runs/proposed/config.yaml --splits val
python tools/report.py --split val
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
python tools/run.py run --config configs/clip.yaml --splits test
python tools/run.py run --config configs/fafa.yaml --splits test
bash scripts/methods/proposed.bash test
python tools/report.py --split test
```

Report sample/case counts alongside such tables. CPU tests use tiny local fixtures;
they are not pretrained validation/test experiments:

```bash
python -m pytest -q
python -m ruff check src tools tests labelstudio
```
