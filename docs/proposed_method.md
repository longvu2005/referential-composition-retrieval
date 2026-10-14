# Proposed method A: shared partial matching without case labels

## Frozen features

The existing person detector supplies full-person boxes. DINO supplies the
letterboxed scene patch grid. A separate frozen FAFA image-only trunk supplies
person crops for BOTH query and target images:

```text
crop -> FAFA visual encoder -> image-only Q-Former hidden tokens -> mean pool h
z_id  = normalize(P_id(h))       # 128 dimensions
z_sem = normalize(P_sem(h))      # 384 dimensions
```

Use the official pinned FAFA checkpoint and preprocessing from
`configs/fafa.yaml`. Pool `extract_features(..., mode="image").image_embeds`,
before native `vision_proj`; never pool projected features or multimodal outputs.
The complete image-only trunk, including Q-Former, is frozen/eval during cache
construction. No instruction enters this cache. `h` is cached, not learned heads.
Mean pooling is a simple person representation, not native FDA scoring.

DINO scene and FAFA person inputs have separate widths and cache metadata. In
`dual`, the scene projection is not on the person identity path. BERT remains
frozen/eval; only its output projection is trained. No FAFA library/model is
loaded during training or retrieval. The DINO directory stores scenes,
detections, identities and DINO crop features for the backbone controls. The
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
Person-token outputs are normalized into `z_comp_si`. Roles remain distinct
in one common composition, without branching on case.

## Coarse retrieval

Keep the existing optimistic soft-identity score and global state branch:

```text
S_id(q,t) = mean_s sum_i normalized(a_si) max_j dot(z_id_qi,z_id_tj)
S_global_state(q,t) = dot(projected change text, projected mean DINO scene)
S_coarse = z_gallery(S_id) + coarse_beta * z_gallery(S_global_state)
```

Population z-scores use common eligible non-self split-gallery support. A
constant branch contributes zero. `identity_only`, `state_only`, empty-person
handling, Top-500 budget, stable ties and complete-gallery ordering are unchanged.
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

## Aligned state, context and missing evidence

Let `pi(i)` denote the assigned target or null. State reads that target only:

```text
e_si = cosine(z_comp_si, z_sem_t,pi(i))
E_si = d_i,pi(i) + e_si                    # matched, main mode
E_si = -missing_penalty                   # null
B_s = sum_i w_si E_si                     # includes null members
B_s = -missing_penalty                   # active Subject with no query member
B = mean_active_subjects B_s
```

Both cosine terms are normalized. `missing_penalty > 2` keeps null below the
minimum two-term score; its default is 2.5. Missing members stay in the pooling
denominator. This is graded evidence, not hard rejection by detector counts.

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
S_fine = softplus(binding_log_scale) * B + C * S_context
```

The scale starts at 1. An entirely unmatched query has zero context contribution
and finite negative binding. Context remains unbounded in A; amplitude limiting
is the separate B extension. Detector misses and grounding mistakes can still
lower positive scores. Thresholds and ranking quality require validation.

## Objective and ranking

```text
L = L_ground + 0.1 L_id + L_ret + L_global_state
```

- Identity: the existing supervised contrastive formulation. Query people and
  valid positive-target people share an ID vocabulary. Unknown IDs, padding,
  duplicate observations and self-pairs are excluded.
- In dual mode, `P_id` receives gradient only from `L_id`. Fine binding detaches
  BOTH identity inputs. With identity loss disabled, AdamW also leaves this head
  untouched. Pose/clothing/background invariance requires suitable cross-image
  supervision and empirical verification.
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

## Architecture controls

`model.representation: shared` is a retrained shared-head control: one identity
head feeds identity loss/coarse, instruction composition and the target-token
identity addition. It requires `model.binding_mode: none`. It uses the same
partial assignment, aligned context and null penalties, while omitting matched
cosine terms. `dual` has independent heads and detached identity binding.
The control does not load an old checkpoint.

DINO/FAFA x shared/dual tests backbone and the complete representation/routing
change. Binding removals compare `both`, `identity`, `semantic`, `none`; they
remove matched score terms, not identity assignment, null penalties, coverage,
semantic context or coarse identity. Evaluate by case and
CandidateRecall@500 as well as ID-mAP/Full-mAP. See [ablations](ablations.md).

## Configuration and checkpoints

Architecture version: `person-id-sem-partial-match-v2`. V1 checkpoints are
rejected before loading weights; retraining is required. Existing compatible
DINO/FAFA caches need no rebuild. Main outputs default to `runs/proposed-a`.

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
