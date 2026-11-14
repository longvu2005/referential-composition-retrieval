# Proposed method: native FAFA composition with relational context

## Frozen features

The Grounding DINO Tiny detector supplies full-person boxes. The frozen vision
half of `openai/clip-vit-base-patch32` supplies the letterboxed 224 x 224 scene:
49 patch tokens of width 768 and the separately projected 512-D global CLS.
Both image and text use revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
The native CLIP text encoder supplies projected token states and a separate
512-D EOT embedding. Text has 77 positions including BOS/EOS; Subject spans are
mapped by character offsets, without adding or freezing random marker embeddings.
All original Subject mentions receive their role injection after the frozen
text backbone. Truncation counts are recorded; losing a complete required Subject
mention raises an explicit error. This does not change FAFA's separate 128-token policy.

Patch and text states enter independent learned context projections to width 384.
Patch tokens are not asserted to be directly aligned to individual text tokens.
Global state bypasses these context projections and uses native pooled embeddings.
State adapters are 512-D affine maps initialized to identity/zero bias; training
retains the same identity-restricted state loss and may adapt the initial CLIP metric.

A separate frozen FAFA image-only trunk supplies
person crops for BOTH query and target images:

```text
crop -> FAFA visual encoder -> image-only Q-Former hidden tokens -> mean pool h
u     = LayerNorm(W_u h + b_u)   # model.dim, separate from scene projection
z_id  = normalize(W_id u + b_id) # 128 dimensions
z_sem = u                       # shared semantic token; no independent head
```

Use the official pinned FAFA checkpoint and preprocessing from
`configs/fafa.yaml`. Pool `extract_features(..., mode="image").image_embeds`,
before native `vision_proj`; never pool projected features or multimodal outputs.
The complete image-only trunk, including Q-Former, is frozen/eval during cache
construction. No instruction enters this cache. `h` is cached, not learned heads.
Mean pooling is a simple person representation, not native FDA scoring.

CLIP scene and FAFA person inputs have separate widths and cache metadata. The
scene projection is independent of the shared person projection. CLIP and its
native projections stay frozen/eval; the additional context projection is trained. No FAFA library/model is
loaded during training or retrieval. The scene directory stores CLIP patches/global
vectors only. A shared detection directory owns ordered pixel boxes and original
image sizes, without GT identity labels. The FAFA directories store crop features
bound to that detection cache ID, independently of CLIP. `GalleryCache` joins these
sources, verifies image order and detection identity, and maps pixel boxes into
the scene encoder's letterbox coordinates at read time.

`GalleryCache.load()` returns `(scene, persons, boxes, mask)` only. Training uses
`TrainSupervision` to align GT train heads to those detections separately and
lazily; validation/test images are rejected by this supervision interface.
Legacy identity metadata is discarded by the cache reader. This changes data
ownership, not the matching geometry or model equations.

## Grounding and composition

Selection text grounds each Subject against scene/context and person semantics:

```text
g_si = Ground(F_q, z_sem_qi, box_qi, selection_s)
a_si = sigmoid(g_si)
```

Coarse uses these independent sigmoid memberships. Fine uses one
exclusive owner per query detection, with a fixed `background_logit`:

```text
p_0i, p_1i, ..., p_Si = softmax_over_subjects(background_logit, g_1i, ..., g_Si)
a_i = 1 - p_0i
owner_i = argmax_s p_si
selected_i = (a_i >= selection_threshold) AND (max_s p_si > p_0i)
m_si = a_i * selected_i * [owner_i = s]
```

Softmax is over Subject/background, not people: a Subject can select any number
of members. No case label or GT member count enters scoring. Subject IDs are
roles in query text. There is no forced top-1 fallback. An empty Subject remains
missing. Fine decisions/weights are detached; raw logits retain masked BCE
supervision. The exclusive distribution conditions the independent odds on at
most one owner; it is not assumed to be a calibrated correctness probability.
The current experiment uses `background_logit=-3.5`; missing fields in older
configs/checkpoints resolve to zero. This constant only enters fine selection,
never the raw grounding BCE logits or coarse sigmoid identity scores. The same
membership drives composition, assignment, context and native fine scoring.

High-IoU NMS suppresses near-duplicate query boxes by grounding confidence and
target boxes by area, with stable detection-order ties. This is a conservative
box heuristic, not an identity oracle. `duplicate_iou=1` disables it. Padding
and missing-annotation masks are applied before supervision. Identity embeddings
remain crop-only.

`StructuredComposition` uses semantic tokens, Subject roles, all Subject-marker
mentions and the complete change text. It keeps one token per Subject/person,
plus CLS/change tokens. Subject summaries are injected into marker tokens, while
individual members remain available. Fine composition receives detached
`logit(m_si)` (unselected slots are masked), giving a `log(m_si)` key prior.
The output is the reference for the Context Reasoner only. It no longer
creates a semantic retrieval embedding. Roles remain distinct without branching
on case.

Every person-semantic input above is the unnormalized `u`,
including both grounding and composition. Target evidence/context also reads
aligned `u`; native FDA uses a separate frozen FAFA representation. Query and target share the
same person projection and LayerNorm. Scene projection remains independent and
can have a different raw input width. No composition residual gate is added.

## Coarse retrieval

Coarse retrieval combines optimistic soft identity with a global state branch:

```text
S_id(q,t) = mean_s sum_i normalized(a_si) max_j dot(z_id_qi,z_id_tj)
S_global_state(q,t) = cosine(A_text E_CLIP_text(change), A_image E_CLIP_image(t))
A_text, A_image start as 512-D identity maps; E_CLIP is frozen.
S_coarse = z_gallery(S_id) + coarse_beta * z_gallery(S_global_state)
```

Population z-scores use common eligible non-self split-gallery support. A
constant branch contributes zero. A target without detected people retains an
identity score of minus infinity. Top-M limits fine scoring, and the remaining
gallery keeps coarse order with stable ties.
Local person semantics feed fine context; they do not replace the global-state
coarse branch.

## One partial identity matching for every fine branch

Normalize membership within each Subject and define assignment importance:

```text
w_si = m_si / sum_i m_si                   # zero for an empty Subject
omega_i = sum_s w_si
d_ij = cosine(stopgrad(z_id_qi), stopgrad(z_id_tj))
X* = argmax_X sum_ij omega_i X_ij (d_ij - identity_threshold)
```

Each query member matches at most one target detection. Each target detection
is used at most once across ALL Subjects. Every member has a private null column
with utility zero. Real edges require `d_ij > identity_threshold`; threshold ties
go to null. SciPy solves the global assignment, not a greedy maximum. Assignment
is detached in both train/retrieve. Semantic scores never change the assignment.
No identity label is used. Normalization balances Subjects of different predicted
sizes, without consulting GT member counts.

## Native composition, aligned FDA and missing evidence

For each public Subject slot and **every** detected query crop:

```text
z_F_si = FAFA.extract_features(crop_qi, condition_s, mode="multimodal")
         .multimodal_embeds
V_F_tj = FAFA.extract_target_features(crop_tj, mode="mean")[0]
a_F_si,j = mean(topk(dot(normalize(z_F_si), normalize(V_F_tj[l])), k=6))
```

The pinned FAFA code performs image/text interaction inside its multimodal
Q-Former, then projects the first text-position fusion token using `text_proj`.
The target path keeps every Q-Former token after `vision_proj` and normalization.
There is no learned projection between these native query and target features.
Target cache uses `cache.storage_dtype`; query cache stays FP32. FDA arithmetic
renormalizes cached features in FP32, including under AMP.

`subject_conditions` retains the complete condition for every public Subject,
without clause splitting or case-specific rules. Mentions of the current Subject
become “the reference person”; other Subject mentions become “the other referenced
person (Subject N)”. Per-crop group wording keeps the existing singular auxiliary
agreement. It never reads case labels, GT identities or member counts. Complete
original change text still enters the Context Reasoner. Independent crop
composition alone does not enforce group/relation correctness. The original FAFA
baseline and `fafa_pooling` retain their separate CPR text adapter.

Let `pi(i)` be the shared identity assignment or null. `retrieval.pooling`
selects the reduction of native FDA evidence; the method remains `proposed`.
The `penalized` reduction is:

```text
E_si = a_F_si,pi(i)                        # matched
E_si = -missing_penalty                    # null
B_s = sum_i w_si E_si                      # includes null members
B_s = -missing_penalty                     # active Subject with no member
B_FAFA = mean_active_subjects B_s
```

`matched_mean` normalizes the matched weighted FDA across active Subjects,
discarding unmatched mass. Subjects with higher matched coverage contribute
more to this global mean. It returns `-missing_penalty` if no evidence matches.

`subject_mean` first normalizes matched FDA within each active Subject, then
averages Subjects equally. A completely unmatched Subject contributes
`-missing_penalty`; unmatched members inside a nonempty Subject do not enter
that Subject's denominator. Thus a present Subject cannot absorb the weight of
an absent Subject. Partial evidence can still miss an identity inside a GROUP;
coverage is a model estimate, not a ground-truth completeness certificate.

All reductions use the same detached identity assignment and grounding weights.
They set the binding offset in training and inference without changing model
parameters or cache features. Identity cosine is not added again. FDA lies in [-1,1], and the null
penalty magnitude remains greater than 1. Empty selection blocks contextual
retrieval gradients, preserving the grounding/matching decision.

`retrieval.pooling` selects the reduction; omitting it retains the `penalized`
fallback for existing configs.

`EvidenceBinding` conditions scene patches on the composed reference, but its
person inputs are gathered using `pi`. Target evidence and geometry are injected
ONLY into the corresponding Subject/person reference token. Null members receive
a learned null token with no scene/geometry update. Other target person crops
never enter the reasoner. Scene remains available through each matched person's
spatially biased attention, including context outside crops.

Directed pair geometry uses `(dx, dy, log width ratio, log height ratio, IoU)` of
matched boxes. Null/unselected pairs have neutral geometry. Two rounds of the
same self-attention layer let member interactions reach CLS. There is no
case-specific relation branch or unrestricted cross-attention to fresh people.
Soft scene attention can still read irrelevant background: this routing is not
a proof of correct semantic ownership.

```text
C_s = sum_i w_si [pi(i) is not null]       # zero for an empty query Subject
C = mean_active_subjects C_s
alpha = softplus(binding_log_scale)
h = tanh(S_context)
S_fine = alpha * B_FAFA + beta * (C * h - (1 - C))
```

The scale is fixed at 1 for new training runs. `beta` is
`retrieval.context_weight`, used by training, inference and mining. The raw
reasoner scalar is unbounded, but its score contribution is bounded. At zero
coverage the residual is exactly `-beta`, independently of the reasoner.
For fixed binding and raw context, the derivative with respect to coverage is
`beta * (tanh(S_context) + 1) >= 0`. This is a conditional monotonicity guarantee,
not a claim that every partially matched image ranks below every complete one.
Coverage is predicted matching support, not a ground-truth identity oracle.
Detector misses and grounding mistakes can still lower positive scores.

## Objective and ranking

```text
L = L_ground + 0.1 L_id + L_coarse + L_context + L_global_state
L_context = L_semantic + context_rejection_weight * L_rejection
```

Each term has one explicit task:

| Term | Score/target | Eligible observations |
| --- | --- | --- |
| `L_ground` | Subject membership logits; balanced BCE | Known query crops, active Subjects with at least one aligned foreground person |
| `L_id` | Normalized person embeddings; SupCon, temperature 0.1 | Deduplicated known-ID crops from queries, Full Positive targets and extra train-only identity observations |
| `L_coarse` | Actual raw `identity_scores`, temperature 0.1 | All-required-ID images are positives, other candidate images are negatives; both query and target need person support |
| `L_context` | Shared bounded fine score, `loss.context_temperature` | Semantic: Full Positives versus same-required-ID negatives. Rejection: Full Positives versus mined wrong/missing-ID negatives. Both ends need nonzero predicted coverage |
| `L_global_state` | Global state cosine, temperature 0.1 | Full Positives versus same-required-ID negatives, independently of person detection coverage |

After `train.sampling.warmup_epochs`, nonzero
`train.identity_sampling.hard_negative_fraction` also admits known-ID crops from
Full-negative candidate images into SupCon. Supplementary labels use the existing
train-only alignment and require exactly one annotated head center in the crop;
ambiguous/unknown/padded crops are excluded without changing BCE alignment.
The label cache is extended before epoch 1; frozen features are reused.

Baseline query/Full Positive observations remain outside the supplemental budget.
Preserve the baseline sampler's additional required-ID positives first (`P`),
then reserve `floor(hard_negative_fraction * (max_crops - P))` slots for hard
target crops. Hardness is the maximum FP32 cosine against unambiguous requested-ID
query anchors with a different GT ID. Multiple Subject slots use actual IDs;
same-ID wrong-condition crops stay identity positives. Random remainder/fallback
comes from the existing sampler with its `crops_per_identity` and
`random_identities` limits. Physical `(image_id, detection_index)` keys are unique
across all sources, while different-image observations of the same ID remain.
Missing valid evidence leaves slots unused; no repetitions are created.

Selection uses detached embeddings, then SupCon gathers the original query/target
embeddings, reusing targets for any sampled crop already in candidates. Only
selected crops outside candidates need the normal extra identity-head forward.
SupCon, temperature, weights and its handling of anchors without positives are
unchanged. Fraction zero (the code default) and the sampling warmup retain the
baseline path. Identity weight zero or a zero supplemental budget skips mining.
Epoch counters record hard, random (including preserved sampled positives), and
total supplemental observations; no per-crop logging is added.

For query q, let P and N be its eligible positive and negative candidate sets:

```text
L_rank(q) = mean_(p in P, n in N) softplus((score[n] - score[p]) / temperature)
L_rank = mean over queries with both P and N nonempty
```

Every Full Positive is trained against every eligible negative. Positives do not
compete against one another, and queries with different positive counts retain
equal weight. Empty supervision yields a finite, differentiable zero. Invalid
entries are removed before arithmetic; non-finite eligible scores are errors.
The pairwise logistic objective follows [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/);
crop identity retains [SupCon](https://arxiv.org/abs/2004.11362). These are ranking
surrogates, not a claim that sampled training loss equals full-gallery mAP.

Grounding averages foreground and background BCE separately within each Subject,
then averages active Subjects per query and valid queries. Unknown IDs/padding
are excluded. A Subject with no aligned foreground crop remains unsupervised.
This avoids weighting a single requested person below a crowd of background
crops. Selection and partial assignment remain predicted and detached; no
teacher-forced membership, identities or group cardinalities enter model scoring.

The same `identity_positive_mask`, derived only from train annotations, serves
three different targets. A same-ID wrong-state image is a **coarse positive**,
but a **semantic-context/state negative**. Only explicitly mined wrong-ID images
receive a rejection label; random wrong-ID fallback images do not. State retains
its identity-restricted supervision.
Candidate sampling still excludes self, all unsampled Full Positives and disputed
negatives from equivalent queries. No validation/test annotation enters a train
loss or negative pool.

The context objective uses the shared `score_context()` path, with the exact
inference matching, coverage gating and reasoner. Context queries lacking a
supported positive/negative pair are skipped. Semantic and rejection losses each
average pairs per eligible query, then average eligible queries independently.
`loss.context_rejection_weight` defaults to zero for omitted old settings; the
current YAML uses `0.2`. This weight is nested under `loss.context_weight`, not a
sixth standalone objective or the inference context coefficient. Rejection is
disabled until the sampling warmup ends. A mined image whose predicted coverage
becomes zero is excluded again in the current training forward. No extra model
forward or classifier is added. `context_active_fraction` retains its semantic
meaning; `context_rejection_active_fraction` is logged separately, alongside
`context_semantic` and `context_rejection`. Global state can still learn when
detections fail. Crop identity and coarse supervision
train the identity head; context gradients train composition, evidence binding,
reasoner and shared projections without differentiating through matching.

Frozen native FDA binding enters the context ranking score without autograd.
The batch carries query/target native FAFA tokens only when context is active.
Pooled FAFA person features remain inputs to learned identity/semantic modules.
`binding_log_scale` is fixed rather than optimized; the optimizer includes only
trainable parameters.

`loss` contains five task weights, one nested rejection weight and four positive
temperatures. `loss.context_temperature` scales the pairwise context loss;
`retrieval.context_weight` sets the residual bound. These are separate from the
task weight `loss.context_weight`. Defaults are starting values, not measured
optima. Pooling and the residual weight must match training during retrieval and
remain fixed during calibration. An untrained state branch cannot enter
training validation with a positive coarse beta.

The default 20-epoch run has a short context warmup:

| Epochs | Active objective |
| --- | --- |
| 1-4 | `L_ground + 0.1 L_id + L_coarse + L_global_state` |
| 5-8 | All five terms; context uses semantic pairs only |
| 9-20 | Fine negative mining and context rejection are also enabled |

Selection and assignment are detached, so unreliable initial grounding/identity
cannot be corrected through context gradients. Warmup first supervises those
components without running composition or fine matching/reasoning. Afterwards,
the base tasks remain active to constrain their shared projections while context
learns. The optimizer, its state and learning rates are retained across phases;
there is no parameter freezing or teacher forcing added at the boundary.

`train.context_warmup_epochs` must be an integer in `[0, train.epochs)`; omitted or
zero means joint training from epoch 1. A positive warmup requires a non-context
objective. When `loss.context_weight=0`, no context phase is scheduled. This is
independent of `train.sampling.warmup_epochs`, which delays hard-negative mining.
The four-epoch default is a starting schedule, not a measured optimum. Compare
against zero warmup with the same total optimizer steps and select on validation.

Full validation and checkpoint selection skip warmup epochs, then follow the
existing epoch interval and always evaluate the final epoch. The selection metric
is unchanged. `last.pt` is still saved every epoch; during warmup it is an
intermediate snapshot. History records `phase` and the effective `context_weight`;
W&B records the effective weight and warmup flag per step and epoch. The saved
configuration retains the final objective weights rather than the temporary zero.

Within Top-M, the final order is the fine-score order. Coarse scores select
candidates and order the tail; they are not fused back into the fine score.
Fine scoring is independent of the number and distribution of other candidates.
All derived ranking arithmetic uses CPU FP32 in inference/replay, while model
forwards retain their configured device and chunking. Small floating-point
forward differences can remain between batch sizes. Stable fine/FAFA ties
retain coarse order; coarse ties retain canonical gallery order.
`best.pt` maximizes `0.5 * overall val Full-mAP + 0.5 * macro-case val Full-mAP`.

### Residual context objective

Training uses the same pairwise score as inference:

```text
h(q,c) = tanh(S_context(q,c))
s(q,c) = alpha * B_FAFA(q,c) + beta * (C(q,c) * h(q,c) - (1 - C(q,c)))
L_context = L_semantic + context_rejection_weight * L_rejection
L_pair = softplus((s_negative - s_positive) / context_temperature)
```

`B_FAFA` uses the same detached identity assignment, `retrieval.pooling`, FDA-k
and missing penalty as inference. FDA is frozen; context gradients reach
composition, evidence binding, the reasoner and shared projections. Eligibility
masks remain unchanged: semantic pairs use identity positives with positive
coverage; rejection pairs use Full Positives and explicitly mined wrong-ID
negatives with positive coverage. Zero-coverage pairs cannot train the reasoner
through a detached coverage gate. Grounding and identity retain their explicit
supervision.

The residual lies in `[-beta, beta]` in both training and inference, including
single-candidate shortlists. Context cannot compensate for an arbitrarily large
binding gap by increasing its raw magnitude. At equal full coverage, its maximum
pairwise correction is `2 * beta`. Choose beta by inspecting binding gaps on
train/validation and retrain with the final value; offline changes are diagnostic
ablations rather than deployment calibration.

The training batch carries `fafa_query` `[B,S,Kq,D]`, `fafa_targets`
`[B,C,Kt,L,D]` and `fafa_pooling` from independent FAFA query/image caches.
The query cache must cover train; this is checked before epoch 1. Warmup skips
native-token batch reads and context computation. History and W&B record
`context_fda_error`, the share of eligible positive/negative pairs that FDA alone
ranks wrong or tied. Matching, sampling, auxiliary losses and checkpoint selection
retain their existing supervision boundaries.

## Configuration and checkpoints

There is one architecture, `fafa-partial-match-v2`. Frozen extraction has
four independent directories: `data.detection_cache`, `data.scene_cache`,
`data.fafa_image_cache` and `data.fafa_query_cache`. The format-7 image index contains packed pooled person means,
native target-token metadata and encoder/crop provenance, with no query text.
One image-only Q-Former forward supplies both hidden means and projected tokens as
defined above. The query index separately
maps sample/Subject slots to content-addressed conditions. Each key binds the
FAFA source/checkpoint/preprocessing, detection crop source, extraction precision,
reference image, public condition and query text policy. Query text edits only
require rerunning `--cache-stage fafa-query`; identical condition features are reused.

FAFA indices contain no scene encoder metadata or scene dimensions. Changing
scene extraction does not invalidate them. Training checkpoints independently
bind the FAFA image cache ID, detection cache ID, scene cache ID and encoder
metadata; retrieval refuses a different scene even when its dimensions match.
Train-only head alignment binds detection identity, independently of appearance.
Completed unified CLIP scene features are readable. Historical boxes require
conversion with the original code revision or a fresh detection build; see
[supported detection inputs](detection_cache.md#supported-detection-inputs).
Build format-7 FAFA caches once in the new independent paths. Their crop-source
identity and preprocessing are part of the provenance contract.
DINO/BERT architecture checkpoints are rejected before loading text weights.
Fresh experiments train from scratch in `runs/proposed`. The architecture
identifier and parameter layout stay unchanged, including the architecture-name
alias. Retrieval requires a checkpoint trained with the bounded residual scorer;
an older checkpoint's parameter layout alone does not establish scoring
compatibility. See [feature contracts](feature_contracts.md).
Offline replay requires a score-schema-3 artifact.
All train/val/test query features are frozen, label-free preprocessing; no
validation/test labels are used to build native features. Training accesses
only train candidates; validation selects checkpoints and coarse shortlist settings.

`checkpoint.max_txt_len` in the native FAFA YAML explicitly selects 128 tokens,
including special tokens, for both the baseline and proposed extraction. The
wrapper sets the caption word cap to at least that limit. Query manifests record
`text_policy`; manifests/checkpoints without it retain the pinned
32-token/50-word meaning. Changing the policy re-extracts query features only.
An interrupted query build must resume with its original policy or use a new
query directory. Query extraction logs and stores `text_stats` for newly extracted
conditions, counting tokenized input before tokenizer truncation. Train/retrieve
check a native YAML when present; cache-only machines can omit it and rely on the
recorded policy and checkpoint contract without installing native FAFA.

`model.fine_matching` defines background, selection/identity thresholds, missing penalty and
duplicate IoU. Settings are restored from the training checkpoint and recorded
in retrieval metadata. Defaults are starting values, not measured optimal values.
Calibration selects among validation artifacts retrieved with different coarse
beta or Top-M; trained pooling and residual weight stay fixed. Changing `model.fine_matching` in a retrieval YAML does not override the checkpoint; use
an explicitly retrained configuration for threshold studies.
New checkpoints save the resolved matching settings, including the actual
background. Retrieval metadata records that effective configuration even if its
input YAML differs. Calibration freezes `selected_background_logit` from the
validation artifact; test rejects a checkpoint background that differs from that
selection (legacy selections default to zero). Changing background requires new
query encoding and fine scoring; score replay cannot change it.

`score_target(..., return_details=True)` exposes matching indices, membership,
Subject/overall coverage, all three binding reductions, raw/gated context and
final score from the same forward pass. Small tests verify structure and
numerics, not benchmark mAP.

Saved rankings include `fafa_rankings` in addition to coarse/fine/final rankings.
This diagnostic sorts FDA binding within the **same Top-M** and retains coarse
order outside it; it is not the full-gallery FAFA + SetMatch baseline.
Every retrieval stores `fafa_binding`, `context` and `coverage` as float32
`[Q,M]` tensors aligned with `coarse_topm`. It also stores `binding_penalized`,
`binding_matched_mean` and `binding_subject_mean` for diagnostic pooling
ablations. Derived diagnostics retain coarse Top-M order. The evaluator reports
`fafa_*` metrics with the same official protocol. Training validation,
validation retrieval and test retrieval fine-score only the Top-M shortlist
through one shared path.

The one `rankings.pt` artifact uses schema version **3**, with query indices,
recorded binding scale, fixed matching policy, scoring configuration and embedded
run/provenance metadata. Artifacts of an earlier schema and checkpoints trained
with previous fine scorers are rejected; train a fresh run using the existing
frozen caches. Each shortlisted pair is fine-scored once per retrieval.
Calibration reads the listed validation artifacts and validation annotations,
without opening any checkpoint, backbone or feature cache. It selects the
artifact with the best validation score; pooling and the residual weight stay
fixed from training and each artifact keeps the coarse beta and Top-M it was
retrieved with. The default `selection_metric` averages overall and macro-case
validation Full-mAP, matching checkpoint selection; `full_map` and
`macro_full_map` are also supported. The first listed artifact wins equal-score
ties. Coarse population normalization uses the complete eligible gallery; fine
scoring has no population normalization. The winning artifact receives full
official evaluation before export.

`candidate_ks` remains fixed independently of Top-M and is evaluated from the
complete coarse order. `selected.yaml` freezes the scoring settings, checkpoint
SHA256 and validation fingerprint; test retrieval rejects changed fingerprints
or scoring settings. Only operational batch sizes may change without selection.
Scoring never uses case labels, identity labels, positives or group cardinality.
See [proposed_runs.md](proposed_runs.md#score-artifacts-and-cpu-replay) for fields,
commands and the limits of reusing fixed model/matching evidence.
