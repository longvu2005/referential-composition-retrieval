# RCR Dataset

This directory contains the data construction pipeline and final benchmark data
for Referential Composition Retrieval (RCR).

RCR retrieves target images by:

1. referring to one or more Subjects in a query image; and
2. specifying the conditions that must hold in the target image.

A Subject is a semantic identifier and may represent one or multiple people.

## Case types

* `INDIVIDUAL`: one Subject containing exactly one identity.
* `GROUP`: one Subject containing two or more identities.
* `DUAL`: two Subjects with independent Subject-specific changes.
* `RELATIONAL`: two Subjects connected by an ordered relation.

`GROUP` is derived only for the one-Subject multi-identity case. A Subject inside
`DUAL` or `RELATIONAL` may contain multiple identities without changing the case.

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
│   │   ├── selection/
│   │   ├── rewrite/
│   │   ├── review/
│   │   └── positives/
│   └── final/
│       └── splits/
└── reports/
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

The supplied canonical export replaces the earlier raw export at the same path.
Subsequent preparation validates and reads it without modifying it.

`data/raw/images` is machine-local; its split subdirectories may be symbolic
links to the actual image tree. The symlinks in the supplied archive point to
a machine-specific path, so repoint them before rebuilding the dataset or
encoding images. Images are not committed to Git. `index.txt` defines the complete
indexed retrieval gallery; `pair_data.json` is source-pair metadata and must not
be used to truncate the Full Positive candidate universe.

`pair_data.json` is also the authoritative source for the benchmark split. For
every derived sample, copy `pairs[*].split` using the ordered key
`(query_image_id, target_image_id)`. Do not infer a split from `batch_name`,
`pool_name`, task order, or a new random partition.

### Working data

`data/work/` contains intermediate, traceable outputs:

* `selection/`: the accepted sample set;
* `rewrite/`: deterministic projections of accepted Stage 2 final text;
* `review/`: human-reviewed and corrected rewrites;
* `positives/`: verified Full Positive sets from the external expansion phase.

Working files are provenance records, not the public benchmark interface.
`rewrite_input.jsonl` and `rewrite_output.jsonl` both use the canonical
`sample_id`, as do all subsequent review and positive files. Old Gemini chunks
and obsolete failed-QC data are removed.

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
* `head_boxes.jsonl`: ground-truth head annotations for evaluation and optional
  alignment of predicted person candidates to identity training labels;
* `manifest.json`: dataset version and counts;
* `splits/*.txt`: sample IDs assigned to each benchmark split.

The proposed method never uses GT head crops or boxes to select detector
candidates or construct identity features. It detects `person` candidates and
projects encoded person crops; a GT head box can provide an identity label for
a matched predicted person during cache construction. Unmatched candidates
have unknown identity `-1` at training time. Evaluation independently derives
identity-positive sets from the same GT identity annotations.

## Construction pipeline

The canonical Stage 2 file has 14 fields per row: `sample_id`, `split`,
`annotator_email`, Query/Target image ID, path and normalized boxes, `case_type`,
`subjects`, and the three final text fields. The 4,311 source rows have already
passed Stage 2 QC and already contain the rewritten instruction.

```bash
bash scripts/phase1_rewrite.bash prepare
```

This validates every source row and ordered pair against `pair_data.json`,
selects all 4,311 records, projects `rewrite_input.jsonl` and
`rewrite_output.jsonl` from the already accepted Stage 2 final text, and
prepares `review_input.jsonl`. It does not call Gemini. Completed human labels
remain in `reviewed.jsonl`; the catalog is available for corrections.
`prompts/rewrite.txt` is the only rewrite prompt/contract in the repository.
The prepared files are traceable source records; the current preparation step
does not invoke Gemini.

After review edits, prepare or refresh the Full Positive catalog:

```bash
python tools/dataset/prepare_handoffs.py positives
```

The 4,311 migrated catalog entries retain the original candidate order and all
completed positive decisions. A reviewed text/subject edit requires inspecting the
corresponding positive label before refreshing; the preparer raises instead of
silently changing a completed positive task. New candidates are drawn from the
complete `index.txt` gallery and filtered by reviewed Subject identities. The
seed target is first, selected and locked. The local UIs are in
`labelstudio/README.md`.

For CLIP ordering of **new** tasks only, use:

```bash
python tools/dataset/prepare_handoffs.py positives --clip-rerank
```

Without this flag, new non-seed candidates use deterministic image-ID order.
Existing tasks and their candidate order are preserved in either mode. CLIP is
not loaded when there are no new reviewed tasks. It requires local images and
downloads its model weights on first use; it does not call Gemini.

The finalization launcher checks and normalizes positive-list ordering. Stop
the positive UI before running it because normalization may rewrite
`positive_sets.jsonl`. The checked-in `final/manifest.json` is version `0.1.0`;
the builder currently defaults to `0.2.0`, so this explicit command produces
a new version after successful finalization:

```bash
bash scripts/phase2_finalize.bash --version 0.2.0
```

The write operation preserves each positive set, including the seed, and orders
it by its candidate catalog. Invalid decisions are rejected before any file is
written. Do not run the finalizer or preparer while the corresponding UI is
serving an older in-memory copy.

The builder checks IDs, reviewed Subject structure, seed and query exclusions,
identity membership, gallery consistency and split membership. It creates
`final_instruction` from `<final_desc>; <final_change>.`. The checked-in
`final/` has **4,311 samples, 37,107 gallery images, version `0.1.0`**, with
split counts 4,033/264/14 for train/val/test. `final_partial/` contains no
materialized dataset in the supplied archive. To generate a partial export as
labels arrive, pass `--allow-partial` to the finalization launcher; it writes
to `final_partial/` and sets `partial: true`. A full build still requires every
selected sample to have reviewed text and positive decisions.

The final CLI also checks completed positive labels against the current reviewed
text/Subjects and original query/seed. It refuses a stale catalog, even if the
positive preparer was skipped after an edit. Human reinspection is required;
this safeguard never silently resets completed labels.

For additional annotations, append canonical Stage 2 rows to the source export
while preserving the existing rows. Every new ordered pair must already be in
`pair_data.json` with its authoritative split. Run phase 1, complete new reviews,
prepare positives, complete new positive decisions, then build final. No review
or positive label is automatically marked complete for a newly appended sample.

## Final sample contract

Each line of `samples.jsonl` is one JSON object:

```json
{
  "sample_id": "...",
  "case_type": "INDIVIDUAL | GROUP | DUAL | RELATIONAL",
  "query_image_id": "...",
  "target_image_id": "...",
  "positive_image_ids": ["..."],
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

`target_image_id` is the canonical seed target from the original pair, while
`positive_image_ids` is the complete reviewed positive set and must contain that
canonical target.

The final builder composes `final_instruction` as
`<final_desc>; <final_change>.` and checks only construction invariants that can be
verified from source metadata: complete inputs, preservation of the seed positive,
valid gallery images, no query image in the positive set, and identity compatibility.
Semantic rewrite quality and exhaustive Full Positive coverage belong to the
human-review and Positive Expansion phases.

## Reproducibility rules

* Keep the canonical source export unchanged after this migration.
* Intermediate outputs are retained.
* Existing samples use accepted final text; no Gemini jobs are needed.
* Human corrections are stored separately from model output.
* Full Positive decisions must be traceable.
* All generated records must use deterministic ordering.
* Rebuild final data through the repository pipeline when the image tree is available.
