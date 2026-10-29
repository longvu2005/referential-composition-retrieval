# Official pretrained FAFA with the RCR adapter

## 1. Architecture and checkpoint

FAFA uses a BLIP-2-based model with frozen EVA-CLIP visual encoder, Q-Former with
32 learned query tokens and normalized projected multimodal query/target features.
Load released `tuned_recall_at1_step.pt` from `configs/fafa.yaml`, with
`model_name=blip2_fafa_cpr`, `model_type=pretrain`. Record checkpoint SHA256 and
missing/unexpected state keys; upstream loading uses `strict=False`.

The [official source](https://github.com/Delong-liu-bupt/Composed_Person_Retrieval/tree/0cc16936f031f7ad166be4cce1be33d0b44b728e/FAFA_SynCPR)
is unchanged and pinned to `0cc16936f031f7ad166be4cce1be33d0b44b728e`.
See [native model/extraction](https://github.com/Delong-liu-bupt/Composed_Person_Retrieval/blob/0cc16936f031f7ad166be4cce1be33d0b44b728e/FAFA_SynCPR/src/lavis/models/blip2_models/blip2_fafa_cpr.py).
`fafa.py` integrates native FAFA; `fafa_adapter.py` implements query Subject
prediction and text routing; `fafa_detector.py` reads shared Grounding DINO Tiny boxes,
and `fafa_matching.py` ports mean Partial SetMatch. These adapters are not official FAFA.

## 2. Inputs, preprocessing and features

FAFA reads the same Grounding DINO Tiny person boxes as Proposed from
`cache/detections` (model `IDEA-Research/grounding-dino-tiny`, box threshold
0.30, text threshold 0.25). Detector inference is not repeated. The shared
cache must have a complete `index.pt`, matching gallery IDs/order and detector
metadata. Its packed index stores pixel boxes and original image sizes, and its
content ID covers the ordered boxes and detector configuration. Legacy scene
boxes must first be explicitly imported with matching detector provenance; the
baseline never depends on CLIP scene features or letterbox resolution.
All eligible boxes retain Proposed's
order and floating-point pixel coordinates: no extra NMS, floor/ceil,
confidence filtering or person cap is applied. Proposed's cache does not
contain detection confidences; score 1.0 means only "already accepted".
Zero detections yield a flagged full-scene fallback for FAFA, and missing
Subject anchors are padded with flagged fallbacks. Fallbacks cannot become
extra group members. The FAFA baseline still builds its own multimodal
query features and FDA scores; the shared cache is boxes only, not a
reuse of Proposed's learned model features.

OpenAI CLIP ViT-B/32 selects crops using `final_desc` clauses, native CLIP
preprocessing and normalized cosine similarity. Hungarian anchors are disjoint.
Every query may add unused crops above similarity 0.20 and within 0.03 of its
assigned Subject anchor, cap 5 members per Subject. Competing Subjects select the
highest eligible similarity. Membership is predicted from image/text similarity;
no `case_type`, GT boxes, identity counts or positives are read. An INDIVIDUAL
annotation does not force one member. Count selector text truncation in metadata.

Reference crop plus condition enters native `extract_features().multimodal_embeds`.
Gallery crops use `extract_target_features(mode="mean")`, retaining target tokens
(upstream's mode argument does not mean-pool them). Use native
`squarepad_transform_test(224, need_size=(384,192))` and eval caption processing.
The baseline and Proposed share `configs/fafa.yaml` with
`checkpoint.max_txt_len=128` and `processors["eval"].max_words=128`.
The upstream 32-token default is explicitly overridden in both workflows.

## 3. Retrieval equations

For normalized component vector u_c and target-crop tokens v_{g,j,l}:

\[
A_{c,j}(g)=\frac{1}{k'}\sum_{l\in\operatorname{TopK}(u_c^\top v_{g,j,l})}
u_c^\top v_{g,j,l},\quad k'=\min(6,L).
\]

This matches native soft FDA inference, without training-temperature division.
`use_soft=false` uses maximum token similarity. `fda_alpha=0.5` is retained in
settings, not an additional inference scoring coefficient. Single-component
scene scoring takes the maximum across target crops. For
multiple components let `K` be the number of query components, `N` the number of
predicted gallery persons, `m=min(K,N)` and `b=-1`. Rectangular Hungarian matches
exactly `m` real pairs maximizing their sum. Final aggregation is the CPR mean:

\[
\pi^*=\arg\max_{\pi}\sum_{(c,j)\in\pi} A_{c,j}(g),\qquad
s(q,g)=\frac{\sum_{(c,j)\in\pi^*}A_{c,j}(g)+(K-m)b}{K}.
\]

This gives partial credit when `K>N`, unlike the legacy max-sum/min adapter.
Real pair matches are mandatory; unmatched penalty is fixed and not tuned.
Rank the full split gallery, excluding self.

## 4. RCR adaptation from text

The adapter reads public Subject slots and text, without a case label.

| Text structure | Behavior |
| --- | --- |
| One explicit clause per Subject, separated by `and Subject N`, each mentioning only its own Subject | Split into Subject-specific conditions |
| Cross-Subject references, missing/duplicate clauses or ambiguous boundaries | Retain the full condition for every Subject |

Independent conditions use the native modification text after removing their
`Subject N` prefix; cross-Subject/ambiguous conditions remain intact for every
Subject, matching CPR Benchmark's `prepare_data.query_texts`. Each predicted
member receives its Subject condition and matches a distinct gallery person.
There is no oracle membership/cardinality or target-box access.

## 5. Hyperparameters and validation

Detector threshold/cap, selector model/threshold/margin/member cap, FDA k/use_soft,
preprocessing and unmatched score are explicit in `configs/fafa.yaml`. Defaults
are initial heuristics, not claimed optima. Compare any manual changes using
complete-val Full-mAP in separate output roots; retain the chosen configuration.
`run --split val` locks configuration/artifact hashes in `protocol.json`.
Test-only rejects missing/stale locks. No automatic FAFA sweep or test-based
hyperparameter selection is performed.

## 6. Training policy

Released pretrained checkpoint, eval/inference mode, no RCR training/fine-tuning.
Native training losses are upstream provenance and not optimized here.
Record `rcr_training=false`.

## 7. Reproduction and caches

Use the isolated FAFA environment in [baselines](../baselines.md):

```bash
.venv-proposed/bin/python tools/run.py build-cache \
  --config configs/proposed.yaml --cache-stage detections
.venv-fafa/bin/python tools/run.py prepare --config configs/fafa.yaml
.venv-fafa/bin/python tools/run.py run --config configs/fafa.yaml --split val
# Once the test protocol is eligible:
.venv-fafa/bin/python tools/run.py run --config configs/fafa.yaml --split test
```

The first command builds the shared detector cache once. FAFA preparation obtains
the pinned source, released checkpoint, CLIP selector and base runtime assets.
Inference blocks network model downloads and fails
clearly on missing assets. It can require substantial RAM/disk/GPU resources.
Read existing DINO detections and cache FAFA gallery/query features and
component scores with input/model provenance. The DINO `cache_id` is included
in FAFA feature cache fingerprints. A warm
feature cache skips FAFA construction. Defaults: gallery fp16, query/scoring
fp32. Save complete config,
FAFA and CLIP checkpoint hashes, DINO cache ID, official FAFA commit, and
membership/fallback/truncation statistics.
Outputs: `runs/fafa/<split>/{scores.npy,rankings.pt,run.json,metrics.json}`.

Gallery cache fingerprints include ordered image IDs, resolved paths, file sizes
and modification times, as in the CLIP baseline. Updating a raw image therefore
invalidates its feature family. Adding this input fingerprint creates a new
FAFA baseline cache directory on the first run after the cleanup; existing files
are retained. Proposed's independent feature caches are unaffected.
Unreadable NPY arrays or incompatible rank/count/dtype are rebuilt atomically.

The CPR text/matching policy remains adapter v3. Changing to the Proposed
detection cache invalidates old detector-specific FAFA gallery feature caches.
Query-text policy and membership changes invalidate query features/scores too.
Old `protocol.json` locks are stale: rerun complete validation before test.
The RCR split-specific gallery and official ID/Full evaluator stay unchanged;
CPR Benchmark's pooled 30,552-image gallery is not transferred.

## 8. Limitations

Detection/CLIP selection errors, extra/missed members and scene fallbacks can
dominate results. Group/relation conditions are approximated independently;
there is no joint scene-relation reasoner, and crop context can be insufficient.
Conditions exceeding 128 FAFA tokens may be truncated. Report the
pinned detector, max-sum/mean Partial SetMatch and missing-person score.
