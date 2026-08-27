# RCR Dataset

This directory contains the data construction pipeline and final benchmark data
for Referential Composition Retrieval (RCR).

RCR retrieves target images by:

1. referring to one or more Subjects in a query image; and
2. specifying the conditions that must hold in the target image.

A Subject is a semantic identifier and may represent one or multiple people.

## Case types

* `SINGLE`: exactly one Subject.
* `MULTI`: two or more Subjects without a primary ordered relation.
* `RELATIONAL`: two or more Subjects connected by an ordered relation.

## Directory structure

```text
dataset/
├── prompts/
│   └── rewrite.txt
├── data/
│   ├── raw/
│   │   ├── annotations/
│   │   ├── metadata/
│   │   └── images
│   ├── work/
│   │   ├── audit/
│   │   ├── selection/
│   │   ├── rewrite/
│   │   ├── review/
│   │   └── positives/
│   └── final/
│       └── splits/
└── reports/
    ├── audit/
    ├── validation/
    └── leakage/
```

## Data layers

### Raw data

`data/raw/` contains immutable source data.

Expected inputs:

```text
data/raw/annotations/export_stage2.jsonl
data/raw/metadata/pair_data.json
data/raw/metadata/index.txt
data/raw/images
```

The source files must never be modified in place.

`data/raw/images` is machine-local and should be a symbolic link to the image
directory. Images are not committed to Git.

### Working data

`data/work/` contains intermediate, traceable outputs:

* `audit/`: optional sample-level audit decisions and notes;
* `selection/`: the selected sample set after the audit;
* `rewrite/`: structured Gemini inputs and sample-level rewrite outputs;
* `review/`: human-reviewed and corrected rewrites;
* `positives/`: verified Full Positive sets from the external expansion phase.

Working files are provenance records, not the public benchmark interface. A new
stage must not overwrite the input of an earlier stage.

### Final data

`data/final/` contains the materialized benchmark data.

Expected outputs:

```text
data/final/samples.jsonl
data/final/images.jsonl
data/final/gallery.jsonl
data/final/head_boxes.jsonl
data/final/manifest.json
data/final/splits/train.txt
data/final/splits/val.txt
data/final/splits/test.txt
```

Their roles are:

* `samples.jsonl`: final RCR samples;
* `images.jsonl`: canonical image registry;
* `gallery.jsonl`: fixed retrieval candidate gallery;
* `head_boxes.jsonl`: ground-truth head annotations for validation and Oracle
  analysis only;
* `manifest.json`: dataset version and counts;
* `splits/*.txt`: sample IDs assigned to each benchmark split.

Ground-truth head boxes must not be used by the main benchmark methods.
Methods requiring localization must use the shared predicted-anchor protocol.

## Construction pipeline

Dataset creation has two main machine phases with two human labeling handoffs.
The machine also prepares the context files consumed by the labeling tool.

```text
raw annotations
    │
    ▼
PHASE 1 — MACHINE / GEMINI
    audit → select → prepare rewrite input → split chunks
    → submit → status → collect → next chunk → merge
    │
    └→ prepare review_input.jsonl
    ▼
[HUMAN] review rewrite → reviewed.jsonl
    │
    ├→ MACHINE: prepare positive_set_input.jsonl
    ▼
[HUMAN] expand Full Positives → positive_sets.jsonl
    │
    ▼
PHASE 2 — MACHINE
    validate handoffs → build final dataset
```

### Phase 1

Prepare the production rewrite once:

```bash
export GEMINI_API_KEY="..."
bash scripts/phase1_rewrite.bash prepare
```

`prepare` runs:

```text
tools/dataset/audit_dataset.py
→ tools/dataset/select_samples.py
→ tools/dataset/prepare_rewrite.py
→ tools/dataset/rewrite_gemini.py split
```

The full structured input remains at:

```text
dataset/data/work/rewrite/rewrite_input.jsonl
```

Only samples that are not already completed or assigned to an unfinished chunk
are split. The default chunk size is 1000:

```text
dataset/data/work/rewrite/chunks/chunk_000001/
├── input.jsonl
├── requests.jsonl
└── manifest.json
```

Use `--chunk-size` when a smaller batch is required:

```bash
bash scripts/phase1_rewrite.bash prepare --chunk-size 800
```

Process chunks sequentially:

```bash
bash scripts/phase1_rewrite.bash submit
bash scripts/phase1_rewrite.bash status
bash scripts/phase1_rewrite.bash collect
```

Then repeat `submit → status → collect` for the next ready chunk. The launcher
refuses to submit a second remote chunk while another one is active. A failed
`batches.create()` keeps the local chunk ready, and an uploaded request file is
reused on retry. If a submitted batch itself ends in a terminal failure, its
samples may be prepared again because no sample-level results were collected.

After every chunk has been collected:

```bash
bash scripts/phase1_rewrite.bash merge
```

`merge` writes:

```text
dataset/data/work/rewrite/rewrite_output.jsonl
dataset/data/work/review/review_input.jsonl
```

The merged rewrite output is cumulative. Existing sample outputs are preserved;
new chunk outputs only fill previously missing IDs. If new raw tasks are added,
run `prepare` again: IDs already processed by Gemini are skipped. This includes
sample-level failures such as blocked, malformed, or otherwise unusable model
responses. Whole failed batch jobs remain retryable because no sample result was
collected from them.

A successful rewrite record is:

```json
{
  "submission_id": "sample_000001",
  "final_desc": "Identify Subject 1 as the man in a black jacket",
  "final_change": "then retrieve target images where Subject 1 is holding a cup"
}
```

If one sample cannot produce a valid rewrite, it still enters `rewrite_output.jsonl`
with empty rewrite fields:

```json
{
  "submission_id": "sample_000002",
  "final_desc": null,
  "final_change": null
}
```

The corresponding reason remains in that chunk's `errors.jsonl` for audit. The
empty fields are passed unchanged into `review_input.jsonl`, where the human
reviewer supplies the rewrite. Such samples are considered processed and are not
sent to Gemini again. The resulting `reviewed.jsonl` must contain non-empty
`final_desc` and `final_change` before the Positive Expansion handoff is prepared.

Human handoffs are also incremental. `review_input.jsonl` excludes IDs already
present in `reviewed.jsonl`. After review, prepare Full Positive labeling with:

```bash
python tools/dataset/prepare_handoffs.py positives
```

`positive_set_input.jsonl` excludes IDs already present in
`positive_sets.jsonl`. Therefore adding new tasks does not send completed old
tasks through Gemini or either human handoff again. The cumulative human files
remain:

```text
dataset/data/work/review/reviewed.jsonl
dataset/data/work/positives/positive_sets.jsonl
```

The Full Positive labeling input contains the reviewed instruction, Query
Subject boxes, and an identity-compatible `candidates` list. A candidate is
included only when it contains every required identity and is not the query
image. The seed target is always first, and `target_image_ids` starts with that
seed.

Local human labeling uses offline Label Studio import/export files under
`labelstudio/`. Rewrite review uses one task per RCR sample. Positive Expansion
groups at most 10 non-seed candidates from the same sample onto one Label Studio
screen, then aggregates all completed groups back into one `positive_sets.jsonl`
record. The grouping exists only in the Label Studio handoff; it does not change
the dataset contract. See `labelstudio/README.md` for the manual workflow.

### Phase 2

After both human files are complete:

```bash
bash scripts/phase2_finalize.bash --version 0.1.0
```

The launcher calls `tools/dataset/build_final.py`. Validation remains in the
Python dataset builder, which checks the handoff IDs, record structure, seed
positive, query exclusion, gallery membership, and required identities before
writing `dataset/data/final/`.

At this stage only, the final instruction is created as:

```text
<final_desc>; <final_change>.
```

### Prompt qualification

Prompt stress tests use the same chunk state machine without rerunning production
audit and selection:

```bash
python tools/dataset/rewrite_gemini.py split --stress-test
python tools/dataset/rewrite_gemini.py submit --stress-test
python tools/dataset/rewrite_gemini.py status --stress-test
python tools/dataset/rewrite_gemini.py collect --stress-test
python tools/dataset/rewrite_gemini.py merge --stress-test
```

## Final sample contract

Each line of `samples.jsonl` is one JSON object:

```json
{
  "sample_id": "...",
  "case_type": "SINGLE | MULTI | RELATIONAL",
  "query_image_id": "...",
  "target_image_ids": ["..."],
  "subjects": [
    {
      "subject_id": 1,
      "identity_ids": ["..."]
    }
  ],
  "final_desc": "...",
  "final_change": "...",
  "final_instruction": "..."
}
```

The final builder composes `final_instruction` as
`<final_desc>; <final_change>.` and checks only construction invariants that can be
verified from source metadata: complete inputs, preservation of the seed positive,
valid gallery images, no query image in the positive set, and identity compatibility.
Semantic rewrite quality and exhaustive Full Positive coverage belong to the
human-review and Positive Expansion phases.

## Reproducibility rules

* Raw data is immutable.
* Intermediate outputs are retained.
* Gemini raw output is never edited in place.
* Human corrections are stored separately from model output.
* Full Positive decisions must be traceable.
* All generated records must use deterministic ordering.
* Final data must only be rebuilt through the repository pipeline.
