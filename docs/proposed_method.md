# Proposed method: native FAFA composition with relational context

## Frozen features

The existing person detector supplies full-person boxes. DINO supplies the
letterboxed scene patch grid. A separate frozen FAFA image-only trunk supplies
person crops for BOTH query and target images:

```text
crop -> FAFA visual encoder -> image-only Q-Former hidden tokens -> mean pool h
u     = LayerNorm(W_u h + b_u)   # model.dim, separate from scene projection
z_id  = normalize(W_id u + b_id) # 128 dimensions
z_sem = u                       # existing semantic API; no independent head
```

Use the official pinned FAFA checkpoint and preprocessing from
`configs/fafa.yaml`. Pool `extract_features(..., mode="image").image_embeds`,
before native `vision_proj`; never pool projected features or multimodal outputs.
The complete image-only trunk, including Q-Former, is frozen/eval during cache
construction. No instruction enters this cache. `h` is cached, not learned heads.
Mean pooling is a simple person representation, not native FDA scoring.

DINO scene and FAFA person inputs have separate widths and cache metadata. The
scene projection is independent of the shared person projection. BERT remains
frozen/eval; only its output projection is trained. No FAFA library/model is
loaded during training or retrieval. The DINO directory stores scenes,
detections, identities and DINO crop features. Training/retrieval use FAFA crops. The
FAFA directory stores only new crop features and its index. `GalleryCache`
combines them and verifies the FAFA index references the same DINO cache ID.

## Grounding and composition

Selection text grounds each Subject against scene/context and person semantics:

```text
g_si = Ground(F_q, z_sem_qi, box_qi, selection_s)
a_si = sigmoid(g_si)
```

Coarse still uses these independent sigmoid memberships. Fine uses one
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
and missing annotation masks are unchanged. Identity embeddings remain crop-only.

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

Keep the existing optimistic soft-identity score and global state branch:

```text
S_id(q,t) = mean_s sum_i normalized(a_si) max_j dot(z_id_qi,z_id_tj)
S_global_state(q,t) = dot(projected change text, projected mean DINO scene)
S_coarse = z_gallery(S_id) + coarse_beta * z_gallery(S_global_state)
```

Population z-scores use common eligible non-self split-gallery support. A
constant branch contributes zero. A target without detected people retains an
identity score of minus infinity. Top-M limits fine scoring, and the remaining
gallery keeps coarse order with stable ties.
Local person semantics do not replace global state/context.

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

Let `pi(i)` be the unchanged identity assignment or null:

```text
E_si = a_F_si,pi(i)                        # matched
E_si = -missing_penalty                    # null
B_s = sum_i w_si E_si                      # includes null members
B_s = -missing_penalty                     # active Subject with no member
B_FAFA = mean_active_subjects B_s
```

Identity cosine is not added again. FDA lies in [-1,1], so the initial penalty
is 1.1 and must exceed 1. This is an uncalibrated starting value. Missing members
stay in the pooling denominator. An empty selection still blocks contextual
retrieval gradients; V1 deliberately preserves grounding/matching behavior.

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
S_fine = softplus(binding_log_scale) * B_FAFA + C * S_context
```

The scale starts at 1. An entirely unmatched query has zero context contribution
and finite negative binding. Context remains unbounded. Detector misses and grounding mistakes can still
lower positive scores. Thresholds and ranking quality require validation.

## Objective and ranking

```text
L = L_ground + 0.1 L_id + L_ret + L_global_state
```

- Identity: the existing supervised contrastive formulation. Query people and
  valid positive-target people share an ID vocabulary. Unknown IDs, padding,
  duplicate observations and self-pairs are excluded.
- `L_id` trains both the person projection and identity
  head. `L_ground` and `L_ret` also train that person projection. Fine still
  detaches both identity inputs; `L_ret` does not directly train the identity
  head. The target context path retains autograd. Native FAFA/FDA is frozen; retrieval
  trains the binding scale and Context Reasoner, including its reference
  composition, person/scene/text projections. It does not train a new composed
  embedding space or backpropagate through selection/assignment. No identity/state
  orthogonality loss or identity removal is introduced.
- Grounding trains raw logits using the existing masked BCE. Retrieval cannot
  lower membership weights to avoid missing penalties. Both losses train shared
  semantic/evidence modules through their respective paths. Retrieval trains
  composition/context using the complete new score and pairwise softplus ranking.
  There is no additional person semantic auxiliary loss in this version.
- Global state keeps same-required-identity positive/negative supervision at image
  level. Wrong-identity images have unknown state labels and are excluded.

Within Top-M:

```text
S_final = z_topM(S_fine) + fine_coarse_weight * z_topM(S_coarse)
```

Weight zero uses raw fine scores (the same ordering). The remaining gallery keeps
coarse order. Periodic evaluation, checkpoint selection, sampling, mining, AMP,
LRU and CPU prefetch retain the existing protocol. `best.pt` maximizes
`0.5 * overall val Full-mAP + 0.5 * macro-case val Full-mAP`.

## Configuration and checkpoints

There is one architecture, `fafa-native-partial-match-v1`. Frozen extraction has
three independent directories: `data.dino_cache`, `data.fafa_image_cache` and
`data.fafa_query_cache`. The format-5 image index contains pooled person means,
native target-token metadata and encoder/crop provenance, with no query text.
One image-only Q-Former forward still supplies both hidden means and projected
tokens; their mathematical definitions are unchanged. The query index separately
maps sample/Subject slots to content-addressed conditions. Each key binds the
FAFA source/checkpoint/preprocessing, DINO crop source, extraction precision,
reference image and public condition. Query text edits only require rerunning
`--cache-stage fafa-query`; unchanged condition features are reused.

Completed format-4 unified caches remain valid image sources without writes.
Replace `data.cache` with `data.fafa_image_cache` and add `data.fafa_query_cache`.
Checkpoints bind image feature identity and encoder provenance, not the query
manifest UUID. Earlier architecture checkpoints still require retraining in a
new run directory; this cache separation does not change the model equations.
All train/val/test query features are frozen, label-free preprocessing; no
validation/test labels are used to build native features. Training accesses
only train candidates; validation selects checkpoints and fusion coefficients.

`model.fine_matching` defines selection/identity thresholds, missing penalty and
duplicate IoU. Settings are restored from the training checkpoint and recorded
in retrieval metadata. Defaults are starting values, not measured optimal values.
Current calibration still selects only coarse/fine fusion coefficients. Changing
`model.fine_matching` in a retrieval YAML does not override the checkpoint; use
an explicitly retrained configuration for threshold studies.

`score_target(..., return_details=True)` exposes matching indices, membership,
Subject/overall coverage, binding, raw/gated context and final score from the same
forward pass. Training logs fine coverage, member count, unmatched-member rate
and empty-selection rate to expose early hard-grounding starvation. Tiny tests
verify structure and numerics, not benchmark mAP.

Saved rankings include `fafa_rankings` in addition to coarse/fine/fused rankings.
This diagnostic sorts FDA binding within the **same Top-M** and retains coarse
order outside it; it is not the full-gallery FAFA + SetMatch baseline.
`fafa_binding`, `context` and `coverage` are aligned with `coarse_topm` in each
saved output. The evaluator reports `fafa_*` metrics with its existing official
protocol. These scores separate embedding quality from context/fusion effects.
