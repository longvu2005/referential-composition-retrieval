# RCR benchmark protocol

This describes the existing evaluator in `src/rcr/evaluation/`. Annotation files
and metric definitions are unchanged. Every method uses the shared evaluator.

## Query, gallery and positives

Inference receives a reference image, `final_desc`, `final_change` and public
`subjects[].subject_id` slots. CLIP uses `final_instruction` by default. Subject
slots identify references within the query; they reveal neither GT identities
nor group cardinality. No method uses `case_type` for retrieval scoring, member
selection or condition decomposition. FAFA uses one predicted-membership rule
for every query and splits only independent clauses identified from text.

GT head boxes, identity labels, seed targets and positives are annotations,
not inference inputs. Proposed may use train annotations for sampling and
supervised losses; CLIP and FAFA are not trained on RCR. Validation annotations
support evaluation and model selection; test annotations support evaluation.
Neither split's labels may determine retrieval scores or group cardinality.
`case_type` remains available for train sampling, provenance and by-case metrics.

The code enforces this boundary with `inference_sample()` and `RCRInputs`:
Subject objects contain only `subject_id`, and annotation fields are absent.
`load_rcr_inputs()` supports feature extraction without head annotations.
`load_rcr_data()` is the annotated interface for training, evaluation and
experiment bookkeeping. Dataset fingerprints and validation locks stay in that
bookkeeping layer; scorers do not receive them as feature inputs.

New visual cache shards contain no GT identities. `GalleryCache.load()` returns
only scene features, person features, predicted boxes and a person mask.
`TrainSupervision` aligns train heads with existing detections in memory, refuses
non-train images, and supplies labels only to batch/loss construction. Detection
diagnostics align GT separately for the requested split. Old cache labels are
ignored by every public cache interface.

The gallery contains **all registered images in the query's PIPA split**, in
canonical `gallery.jsonl` order. Membership comes from `images.jsonl` paths, not
annotated pairs. Every ranking contains each gallery image once, except the
query image. Baseline score ties retain canonical gallery order. `leftover`
images never enter a query's gallery.

For gallery G(q), union of required Subject identities I(q), and annotated
identities A(g) of image g:

\[
P_{ID}(q)=\{g\in G(q)\setminus\{q\}:I(q)\subseteq A(g)\}.
\]

ID positives require **all** identities; extra identities are allowed. They
ignore condition, Subject order and grouping. Full positives are reviewed
`positive_image_ids` intersected with G(q) in memory by the loader. Stored
annotations are not edited. All Full positives must be ID positives. Empty
positive sets and incomplete/duplicate/self-inclusive rankings fail evaluation.

## Metrics and paper tables

For positive set P, ranked relevance y_j and H_j=sum(y_1..y_j):

\[
AP(q)=\frac{1}{|P|}\sum_{j=1}^{|r|}y_j\frac{H_j}{j},\qquad
R@K(q)=\mathbf{1}[P\cap r_{1:K}\ne\varnothing].
\]

The existing R@K is **query-level success**, not the fraction of recovered
positives. mAP/R@K are arithmetic means over queries. By-case means use only
that case. Overall is query weighted, not the unweighted mean of case means.

| Table | Metrics |
| --- | --- |
| Overall | ID-mAP, ID-R@1/5/10, Full-mAP, Full-R@1/5/10 |
| By-case | ID-mAP, ID-R@1, Full-mAP, Full-R@1 for each case |

Case order is INDIVIDUAL, GROUP, DUAL, RELATIONAL. An absent case is blank/—/--,
never a measured zero. Coarse/fine and CandidateRecall are Proposed diagnostics
and excluded from these tables. CandidateRecall@K is the fraction of Full
positives retained in the coarse shortlist, distinct from R@K.

## Split and leakage policy

Use the same final export and ordered gallery for every method. Only train may
train model parameters. Validation selects hyperparameters/checkpoints; CLIP
fusion must select by complete-validation Full-mAP. FAFA adapter settings are
manually chosen on validation, then frozen. Neither baseline is fine-tuned on
RCR. Never choose settings, checkpoints, case rules or methods from test labels,
metrics or examples. Reporting performs no selection or inference.

Baseline `run` defaults to validation only. CLIP test requires matching
`tuning.json`; FAFA test requires matching `protocol.json`, covering validation,
pretrained artifacts and adapter settings. These locks do not use test labels.
Low-level `retrieve` remains available for fixed-weight/smoke diagnostics;
unselected fusion and unfrozen FAFA test results are rejected in paper reports.

Version 0.1.0 currently contains:

| Split | Queries | Gallery | INDIVIDUAL | GROUP | DUAL | RELATIONAL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 4,033 | 17,000 | 3,472 | 137 | 269 | 155 |
| val | 264 | 5,684 | 194 | 12 | 36 | 22 |
| test | 14 | 7,868 | 5 | 2 | 3 | 4 |

The **14-query test is only a pipeline-validation split**, not a publication-scale
benchmark. Final publication testing needs a sufficiently large, independently
reviewed export and fresh validation locks. This work does not change annotations.

## Reproducibility and saved-run compatibility

New `run.json` records include effective config, checkpoint SHA256, dataset
version/content digest, split fingerprint, ordered sample/gallery digests,
query/gallery/case counts, protocol/self-exclusion, Git commit/dirty state,
Python/platform/package versions, device and elapsed time. Encoder precision,
cache identity and truncation/fallback counts are recorded by each baseline.
FAFA also saves its official commit and detector/selector hashes. Use a clean
committed checkout: a dirty flag cannot reconstruct uncommitted code.

Dataset/split digests describe labels for provenance only. Baseline scoring and
feature caches use image/text inputs and model artifacts, not positive or identity
labels. The full-dataset digest is never used as a tuning key.

Evaluation binds `metrics.json` to the exact `run.json` SHA256. Reporting checks
that binding, split fingerprint, counts, sample order, case coverage, per-query
aggregation and available dataset/gallery metadata across methods. Method
checkpoints/configs/runtime may differ by design. Legacy runs need their original
split fingerprint and saved-ranking re-evaluation to attach the JSON binding;
missing historical Git/runtime details are not invented. Results lacking an
original split fingerprint cannot be certified by the reporter.

Use a distinct `output.dir` for each scientific experiment. Training refuses a
root containing `config.yaml`, `best.pt` or `last.pt` before loading data/models.
Retrieval, evaluation, calibration and reports replace their active outputs;
there is no automatic archive. Publishing new rankings invalidates old metrics
and metadata first; evaluate again after a successful retrieval.
