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
loaded during training or retrieval. The scene directory stores CLIP patches/global vectors and
predicted crop geometry, without GT identity labels.
Training/retrieval use FAFA crops. The
FAFA directory stores crop features and its index. `GalleryCache`
combines them and verifies the FAFA index references the same scene cache ID.

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
exclusive owner per query detection, with a fixed background logit of zero:

```text
p_0i, p_1i, ..., p_Si = softmax_over_subjects(0, g_1i, ..., g_Si)
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

`subject_conditions` splits a change only when each explicit Subject clause
mentions its own Subject and every Subject has a clause. Ambiguous/group/relation
text is retained, with the current role expressed as “the reference person”.
It never reads case labels, GT identities or member counts. Complete original
change text still enters the Context Reasoner. Independent crop composition
alone does not enforce group/relation correctness.

Let `pi(i)` be the shared identity assignment or null:

```text
E_si = a_F_si,pi(i)                        # matched
E_si = -missing_penalty                    # null
B_s = sum_i w_si E_si                      # includes null members
B_s = -missing_penalty                     # active Subject with no member
B_FAFA = mean_active_subjects B_s
```

Identity cosine is not added again. FDA lies in [-1,1], so the initial penalty
is 1.1 and must exceed 1. This is an uncalibrated starting value. Missing members
stay in the pooling denominator. An empty selection blocks contextual retrieval
gradients, preserving the grounding/matching decision.

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
S_fine_raw = alpha * B_FAFA + C * S_context
```

The scale starts at 1. An entirely unmatched query has zero context contribution
and finite negative binding. Context remains unbounded. Detector misses and
grounding mistakes can still lower positive scores. Thresholds and ranking quality
require validation.

## Objective and ranking

```text
L = L_ground + 0.1 L_id + L_ret + L_global_state
```

- Identity: supervised contrastive learning. Query people and
  valid positive-target people share an ID vocabulary. Unknown IDs, padding,
  duplicate observations and self-pairs are excluded. A bounded train-only crop
  sample adds cross-image observations for query IDs and other train IDs without
  entering retrieval scoring.
- `L_id` trains both the person projection and identity
  head. `L_ground` and `L_ret` also train that person projection. Fine
  detaches both identity inputs; `L_ret` does not directly train the identity
  head. The target context path retains autograd. Native FAFA/FDA is frozen; retrieval
  trains the binding scale and Context Reasoner, including its reference
  composition, person/scene/text projections. It does not train a new composed
  embedding space or backpropagate through selection/assignment. No identity/state
  orthogonality loss or identity removal is introduced.
- Grounding trains raw logits using masked BCE. Retrieval cannot
  lower membership weights to avoid missing penalties. Both losses train shared
  semantic/evidence modules through their respective paths. Retrieval trains
  composition/context using the complete fine score and pairwise softplus ranking.
  There is no additional person semantic auxiliary loss in this version.
- Global state keeps same-required-identity positive/negative supervision at image
  level. Wrong-identity images have unknown state labels and are excluded.

Within Top-M:

```text
gated_context = C * S_context
z(x) = (float32(x) - mean(x)) / max(std(x, unbiased=False), 1e-6)
S_fine = alpha * B_FAFA + 0.0325 * z_topM(gated_context)
S_final = z_topM(S_fine) + fine_coarse_weight * z_topM(S_coarse)
```

`L_ret` continues to use `S_fine_raw` from `score_target()`. Training batches do
not define inference normalization groups. Periodic validation, standalone
retrieval and calibration all use `retrieval.fine_score_mode: context_zscore`
with `context_weight: 0.0325` by default. Context normalization happens **after**
coverage gating and after all Top-M candidates have been scored, in original
Coarse order. Each query has independent statistics. A constant vector, zero
coverage or M=1 contributes zero normalized context. NaN/Inf fine components
and FP32 arithmetic overflow raise explicit errors.

Ranking arithmetic and stable shortlist sorting run on CPU FP32, shared with
offline replay. Model forwards retain their configured device/chunking. Small
floating point differences between differently sized model-forward batches may
remain; changing batch sizes never changes the normalization population.

Final fusion normalizes both branches on common **finite Coarse** support. A
positive fusion weight leaves unsupported candidates at minus infinity; weight
zero uses `S_fine` directly and preserves the previous policy of ignoring Coarse
support. The context statistics above always use the entire Top-M. These are two
distinct normalization steps. Fine/FAFA/fused ties retain Coarse order; Coarse
ties retain canonical gallery order. The remaining gallery keeps coarse order.

`retrieval.fine_score_mode: legacy_raw` uses `S_fine_raw` as the fine ranking score
and ignores `context_weight`. No architecture, weights, losses, optimizer or
`ARCHITECTURE_VERSION` changes are required; old compatible checkpoints load
their learned `binding_scale` normally. Periodic evaluation, candidate sampling, AMP
and bounded cache reads use the shared experiment protocol. `best.pt` maximizes
`0.5 * overall val Full-mAP + 0.5 * macro-case val Full-mAP`.

## Configuration and checkpoints

There is one architecture, `clip-b32-fafa-partial-match-v2`. Frozen extraction has
three independent directories: `data.scene_cache`, `data.fafa_image_cache` and
`data.fafa_query_cache`. The current format-6 image index contains packed pooled person means,
native target-token metadata and encoder/crop provenance, with no query text.
One image-only Q-Former forward supplies both hidden means and projected tokens as
defined above. The query index separately
maps sample/Subject slots to content-addressed conditions. Each key binds the
FAFA source/checkpoint/preprocessing, scene crop source, extraction precision,
reference image, public condition and query text policy. Query text edits only
require rerunning `--cache-stage fafa-query`; identical condition features are reused.

Build new scene, FAFA image and FAFA query caches in the separate CLIP paths.
Their crop-source identity and preprocessing are part of the provenance contract.
DINO/BERT architecture checkpoints are rejected before loading text weights.
Train the new modules from scratch in `runs/proposed-clip-b32`; this is not a
weight-compatible checkpoint conversion. The evaluator and offline replay still
read historical ranking artifacts using their recorded score schemas.
All train/val/test query features are frozen, label-free preprocessing; no
validation/test labels are used to build native features. Training accesses
only train candidates; validation selects checkpoints and fusion coefficients.

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

`model.fine_matching` defines selection/identity thresholds, missing penalty and
duplicate IoU. Settings are restored from the training checkpoint and recorded
in retrieval metadata. Defaults are starting values, not measured optimal values.
Calibration selects only coarse/fine fusion coefficients. Changing
`model.fine_matching` in a retrieval YAML does not override the checkpoint; use
an explicitly retrained configuration for threshold studies.

`score_target(..., return_details=True)` exposes matching indices, membership,
Subject/overall coverage, binding, raw/gated context and final score from the same
forward pass. Small tests verify structure and numerics, not benchmark mAP.

Saved rankings include `fafa_rankings` in addition to coarse/fine/fused rankings.
This diagnostic sorts FDA binding within the **same Top-M** and retains coarse
order outside it; it is not the full-gallery FAFA + SetMatch baseline.
`fafa_binding`, `context` and `coverage` are aligned with `coarse_topm` in each
saved output. The evaluator reports `fafa_*` metrics with the shared official
protocol. These scores separate embedding quality from context/fusion effects.

Score schema 1 also stores the actual Coarse scores and complete fine/fused score
components for every computed shortlist candidate; all `[N,M]` fields share
Coarse Top-M order. Per-query context mean/std, the learned binding scale and
normalization/fusion configuration make rankings reproducible on CPU. Calibration
reuses raw query-target components across weight pairs; it recomputes context
statistics for each distinct Coarse shortlist, never across queries or chunks.
Only `coarse_beta` and `fine_coarse_weight` enter the validation grid. Context
mode/weight remain fixed and are recorded in `selected.yaml`, together with the
selected pair, Top-M, checkpoint SHA256 and validation fingerprint. Retrieval
rejects modifications to the selected scoring settings; batch sizes may change.

For post-training studies that also change Coarse shortlist membership,
`retrieve/run --save-tuning --splits val` exports raw Coarse identity/state and
Fine binding/context/coverage scores on the complete split gallery. Each
non-self pair is Fine-scored once. This explicit full-gallery pass is separate
from periodic training validation and stores its fixed matching policy and
provenance inside `rankings.pt`. CPU replay can then recompute beta, requested
Top-M, context weight and final fusion without loading a model or feature cache.
It recomputes Coarse eligibility/population statistics and the complete Coarse
order before normalizing the new Fine shortlist. It never uses case labels,
identity labels, positives or group cardinality for scoring. See
[proposed_runs.md](proposed_runs.md#full-gallery-tuning-export) for the schema and
limitations; exported scores cannot change the underlying matching or model.
