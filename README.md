# Referential Composition Retrieval

This repository contains the dataset construction pipeline, benchmark methods,
and shared evaluation protocol for Referential Composition Retrieval (RCR).

## Task

Given:

1. a query image;
2. a referential description specifying who must be selected; and
3. a target condition specifying what must hold in the desired image;

the goal is to rank gallery images that satisfy both identity preservation and
the requested target condition.

An RCR query may refer to one person, multiple people, or an ordered relation
between multiple Subjects.

## Case types

* `SINGLE`: exactly one Subject.
* `MULTI`: two or more Subjects without a primary ordered relation.
* `RELATIONAL`: two or more Subjects connected by an ordered relation.

A Subject is a semantic identifier and may represent one or multiple people.

## Repository structure

```text
referential-composition-retrieval/
├── dataset/              dataset files, prompt, and reports
├── labelstudio/          local review UI + offline positive handoff
├── src/rcr/
│   ├── dataset/          dataset construction logic
│   ├── methods/          retrieval methods
│   ├── evaluation/       shared benchmark protocol and metrics
│   └── utils/            shared utilities
├── tools/                Python command-line entrypoints
├── scripts/              Bash launchers
├── configs/              dataset, method, evaluation, and experiment configs
├── tests/                dataset, method, and evaluation tests
├── cache/                generated detections and features
├── checkpoints/          shared and method-specific checkpoints
├── runs/                 experiment outputs
├── tables/               aggregated benchmark tables
└── docs/                 additional documentation
```

## Code organization

The repository follows three main layers:

```text
scripts/**/*.bash
    → tools/*.py
        → src/rcr/*
```

* `scripts/` contains shell launchers only.
* `tools/` parses command-line arguments and invokes repository code.
* `src/rcr/` contains all reusable Python implementation.
* Source code must not import from `tools/` or `scripts/`.

The `dataset`, `methods`, and `evaluation` scopes are kept separate throughout
the repository.

## Dataset

Dataset construction is documented in [`dataset/README.md`](dataset/README.md).

The pipeline has two machine phases separated by human annotation:

```text
PHASE 1 — MACHINE / GEMINI
  audit → select → prepare rewrite input → split chunks
  → submit → status → collect → next chunk → merge
  → prepare review labeling input

[HUMAN + HANDOFF PREP]
  review rewrite → prepare positive-set labeling input → expand positive sets

PHASE 2 — MACHINE
  validate handoffs → build final dataset
```

Run Phase 1:

```bash
export GEMINI_API_KEY="..."
bash scripts/phase1_rewrite.bash prepare
bash scripts/phase1_rewrite.bash submit
bash scripts/phase1_rewrite.bash status
bash scripts/phase1_rewrite.bash collect
# repeat submit → status → collect until no ready chunk remains
bash scripts/phase1_rewrite.bash merge
```

`prepare` creates fixed chunks (1000 samples by default). Only one chunk may be
active on Gemini at a time. `merge` creates `rewrite_output.jsonl` and the
incremental `review_input.jsonl`. A sample-level Gemini failure is preserved as an
empty rewrite for human review and is treated as processed, so it is never
resubmitted. Whole failed batch jobs remain retryable. Rewrite review runs through the
repo-local UI, while Full Positive selection uses offline Label Studio
import/export files. See [`labelstudio/README.md`](labelstudio/README.md). No
Label Studio API client is used.
After producing
`reviewed.jsonl`, prepare the positive-set labeling input with:

```bash
python tools/dataset/prepare_handoffs.py positives
```

After `positive_sets.jsonl` is complete, run Phase 2:

```bash
bash scripts/phase2_finalize.bash --version 0.1.0
```

`final_instruction` is created only when the final dataset is built.

## Methods

Methods are organized under `src/rcr/methods/`:

* `common/`: shared adaptations and components;
* `simple/`: simple and diagnostic baselines;
* `published/`: adaptations of published retrieval methods;
* `proposed/`: the proposed RCR method.

Shared adaptations such as SetMatch and Predicted Anchor must be implemented
once under `common/` and reused consistently by every applicable method.

A method must not modify the dataset, gallery, positive sets, evaluation
metrics, or shared localization protocol.

## Evaluation

All methods are evaluated using the same gallery and the same final ranking.

The benchmark reports:

### Identity retrieval

* ID-mAP
* ID-R@1
* ID-R@5
* ID-R@10

### Full RCR retrieval

* Full-mAP
* Full-R@1
* Full-R@5
* Full-R@10

Full-mAP is the primary benchmark metric.

For multi-Subject queries, a Full Positive must satisfy all required Subjects
and conditions. Partial matches do not count as Full Positives.

The query image must not be evaluated as one of its own gallery candidates.

Ground-truth boxes are reserved for validation and Oracle analysis. The main
benchmark uses the shared predicted-anchor protocol whenever localization is
required.

## Setup

Create and activate the Conda environment:

```bash
conda create -n rcr python=3.11 pip -y
conda activate rcr
```

Install the project for development:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -c "import rcr; print(rcr.__file__)"
```

## Reproducibility

The following rules apply to all official experiments:

* raw data is immutable;
* generated artifacts are separated from source code;
* intermediate dataset decisions remain traceable;
* configs used for official runs are preserved;
* random behavior must be explicitly seeded;
* gallery and positive definitions are fixed across methods;
* shared detector settings and caches are fixed across applicable methods;
* method outputs are written under `runs/`;
* final tables are built from saved run artifacts;
* evaluation failures must not be silently ignored.

## Project status

The repository is currently in the dataset reconstruction stage.

Dataset, method, and evaluation entrypoints will be added incrementally. A
component must not be described as runnable until its implementation and tests
are present.
# referential-composition-retrieval
