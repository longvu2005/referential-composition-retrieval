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
`fafa.py` integrates native FAFA; `fafa_adapter.py` implements RCR localization,
Subject prediction and SetMatch. These adapter components are not official FAFA.

## 2. Inputs, preprocessing and features

Faster R-CNN ResNet50-FPN-v2 with COCO weights predicts people on RGB scenes.
Keep up to 10 candidates, threshold 0.55, and lower-score candidates if required
for public Subject slots (query) or one crop (gallery). No detections cause a
whole-scene fallback; fallback crops never become extra group members.

OpenAI CLIP ViT-B/32 selects crops using `final_desc` clauses, native CLIP
preprocessing and normalized cosine similarity. Hungarian anchors are disjoint.
Non-INDIVIDUAL cases may add unused crops above similarity 0.20 and within 0.03
of a Subject's best score, cap 5 members per Subject. Competing Subjects select
the highest eligible similarity. Group cardinality is predicted; no GT boxes,
identity counts or positives are read. Count selector text truncation in metadata.

Reference crop plus condition enters native `extract_features().multimodal_embeds`.
Gallery crops use `extract_target_features(mode="mean")`, retaining target tokens
(upstream's mode argument does not mean-pool them). Use native
`squarepad_transform_test(224, need_size=(384,192))` and eval caption processing.
The pinned pretrain config resolves max text length to 32; native truncation may
shorten long conditions.

## 3. Retrieval equations

For normalized component vector u_c and target-crop tokens v_{g,j,l}:

\[
A_{c,j}(g)=\frac{1}{k'}\sum_{l\in\operatorname{TopK}(u_c^\top v_{g,j,l})}
u_c^\top v_{g,j,l},\quad k'=\min(6,L).
\]

This matches native soft FDA inference, without training-temperature division.
`use_soft=false` uses maximum token similarity. `fda_alpha=0.5` is retained in
settings, not an additional inference scoring coefficient. Single-component
scene scoring takes the maximum across target crops. Multiple components pad
insufficient crops with `unmatched_score=-1`, then use:

\[
\pi^*=\arg\max_{\pi\text{ injective}}\sum_c A_{c,\pi(c)}(g),\qquad
s(q,g)=\min_c A_{c,\pi^*(c)}(g).
\]

Hungarian maximizes the **sum**, then aggregation takes the **minimum**; assignment
is not bottleneck-optimal. Rank the full split gallery, excluding self.

## 4. RCR adaptation by case

| Case | Behavior |
| --- | --- |
| INDIVIDUAL | One CLIP-selected member; maximum FDA across target people |
| GROUP | Predicted members receive intact group condition; SetMatch |
| DUAL | Split clear `and Subject N` clauses, otherwise retain full condition |
| RELATIONAL | Retain full relation for each Subject; preserve roles in text |

Own Subject mentions become `the reference person`; other Subject mentions remain
explicit other-person references. Each component matches a different target crop.
There is no oracle membership/cardinality or target-box access.

## 5. Hyperparameters and validation

Detector threshold/cap, selector model/threshold/margin/member cap, FDA k/use_soft,
preprocessing and unmatched score are explicit in `configs/fafa.yaml`. Defaults
are initial heuristics, not claimed optima. Compare any manual changes using
complete-val Full-mAP in separate output roots; retain the chosen configuration.
`run --splits val` locks configuration/artifact hashes in `protocol.json`.
Test-only rejects missing/stale locks. No automatic FAFA sweep or test-based
hyperparameter selection is performed.

## 6. Training policy

Released pretrained checkpoint, eval/inference mode, no RCR training/fine-tuning.
Native training losses are upstream provenance and not optimized here.
Record `rcr_training=false`.

## 7. Reproduction and caches

Use the isolated FAFA environment in [baselines](../baselines.md):

```bash
.venv-fafa/bin/python scripts/run.py prepare --config configs/fafa.yaml
.venv-fafa/bin/python scripts/run.py run --config configs/fafa.yaml --splits val
# Once the test protocol is eligible:
.venv-fafa/bin/python scripts/run.py run --config configs/fafa.yaml --splits test
```

Preparation obtains pinned source, released checkpoint, detector, CLIP selector
and base runtime assets. Inference blocks network model downloads and fails
clearly on missing assets. It can require substantial RAM/disk/GPU resources.
Cache detection, gallery/query features and component scores with input/model
provenance. Defaults: gallery fp16, query/scoring fp32. Save complete config,
three checkpoint hashes, official commit, membership/fallback/truncation statistics.
Outputs: `runs/fafa/<split>/{scores.npy,rankings.pt,run.json,metrics.json}`.

## 8. Limitations

Detection/CLIP selection errors, extra/missed members and scene fallbacks can
dominate results. Group/relation conditions are approximated independently;
there is no joint scene-relation reasoner, and crop context can be insufficient.
Short native text context may truncate conditions. The adapter fallback and
max-sum/min SetMatch conventions must be reported explicitly.
