# RCR Dataset

The dataset supports retrieval from a reference image, Subject descriptions and
a requested change. A Subject may refer to one or several people.

## Case types

| Case | Description |
| --- | --- |
| `INDIVIDUAL` | One Subject with one identity |
| `GROUP` | One Subject with multiple identities |
| `DUAL` | Two Subjects with separate requested changes |
| `RELATIONAL` | Two Subjects connected by an ordered relation |

## Data layout

```text
dataset/data/raw/{annotations,metadata,images}/
dataset/data/work/{selection,review,positives}/
dataset/data/final/
```

Raw inputs include `annotations/export_stage2.jsonl`, `metadata/pair_data.json`,
`metadata/index.txt` and the local image tree. Preserve canonical source records.
Ordered pairs in `pair_data.json` define benchmark sample splits; `index.txt`
defines the indexed image gallery. Raw images are stored separately from Git.

Working data records selection, text review and Full Positive decisions.
Records use a stable `sample_id` throughout the pipeline.

## Final data

| File | Contents |
| --- | --- |
| `samples.jsonl` | Queries, Subjects, instructions and reviewed positives |
| `images.jsonl` | Canonical image registry |
| `gallery.jsonl` | Ordered image registry for shared features |
| `head_boxes.jsonl` | Ground truth head and identity annotations |
| `manifest.json` | Dataset version and counts |
| `splits/{train,val,test}.txt` | Sample IDs for each split |

Retrieval uses the query's image split and excludes the reference image.
The loader intersects reviewed positives with that gallery in memory.
See the [benchmark protocol](../docs/benchmark_protocol.md) for evaluation rules.

## Construction pipeline

Install the dataset environment and prepare text review tasks:

```bash
bash scripts/setup.bash dataset
bash scripts/data/prepare_review.bash
```

Complete text and Subject review using the
[annotation UI](../labelstudio/README.md), then prepare positive candidates:

```bash
bash scripts/data/prepare_positives.bash
```

Complete Full Positive decisions, stop the annotation UI and build the export:

```bash
bash scripts/data/finalize.bash --version 0.1.0
```

Use the intended dataset version when publishing an updated export.
`--allow-partial` writes an incomplete labeling export to `final_partial/`.
Preparation preserves existing decisions; changed text or Subjects require
reinspection of affected positive labels. Stop the relevant UI before refreshing
or normalizing its files.

Optional CLIP ordering for new positive tasks uses the `dataset-clip` environment
and `prepare_positives.bash --clip-rerank`. Existing tasks retain their order.
Individual pipeline commands are available under `tools/data/`.

## Sample fields

| Field | Meaning |
| --- | --- |
| `sample_id`, `case_type` | Stable query identifier and case |
| `query_image_id`, `target_image_id` | Reference image and original seed target |
| `positive_image_ids` | Reviewed Full Positives, including the seed target |
| `subjects` | Subject slots and annotated identity membership |
| `final_desc`, `final_change`, `final_instruction` | Reviewed text and composed instruction |

Construction validates identifiers, split membership, seed preservation and
identity compatibility. Text quality and positive completeness are established
through human review. Use the [audit guide](../docs/full_positive_audit.md) for
integrity and review-coverage checks.
