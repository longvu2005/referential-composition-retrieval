# Proposed method: separate person identity and semantics

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
`configs/methods/fafa.yaml`. Pool `extract_features(..., mode="image").image_embeds`,
before native `vision_proj`; never pool projected features or multimodal outputs.
The complete image-only trunk, including Q-Former, is frozen/eval during cache
construction. No instruction enters this cache. `h` is cached, not learned heads.
Mean pooling is a simple person representation, not native FDA scoring.

DINO scene and FAFA person inputs have separate widths and cache metadata. In
`dual`, the scene projection is not on the person identity path. BERT remains
frozen/eval; only its output projection is trained. No FAFA library/model is
loaded during training or retrieval.

## Grounding and composition

Selection text grounds each Subject against scene/context and person semantics:

```text
g_si = Ground(F_q, z_sem_qi, box_qi, selection_s)
a_si = sigmoid(g_si)
```

Independent sigmoid memberships support GROUP. Padding and missing annotated
identities retain the existing loss masks. Selection text chooses people;
identity embeddings themselves remain crop-only.

`StructuredComposition` uses semantic tokens, Subject roles, all Subject-marker
mentions and the complete change text. It keeps one token per Subject/person,
plus CLS/change tokens. Subject summaries are injected into marker tokens, while
individual members remain available. Reference keys retain the existing
log-sigmoid membership prior. The person-token outputs are normalized to form
`z_comp_si`. DUAL roles stay distinct and RELATIONAL can read both Subjects.

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

## Fine binding and context

Identity and requested semantics meet on the SAME target person:

```text
S_id_ij  = dot(stopgrad(z_id_qi), stopgrad(z_id_tj))
S_sem_sij = dot(z_comp_si, z_sem_tj)
M_sij = S_id_ij + S_sem_sij
B(q,t) = mean_active_subjects sum_i normalized(a_si) max_j M_sij
```

The maximum is taken AFTER summing pairwise scores. Separately maximizing ID
and semantics would let two different target people satisfy the two conditions.
This first version allows target reuse; it does not guarantee one-to-one matching
or complete GROUP coverage. Missing people contribute zero binding evidence.

The existing shared `EvidenceBinding` still conditions scene patches, then binds
scene/geometry evidence to target semantic person features. In dual mode,
`TargetPersonBuilder` combines bound evidence (which already has a semantic
residual) with geometry, without an identity addition. The existing set reasoner
produces a context score for actions/context/relations outside individual crops:

```text
S_fine_new = S_context + softplus(binding_log_scale) * B
```

The scale starts at 1 and is trained by retrieval loss. Identity/semantic pair
coefficients are fixed at 1. Identity and semantic representations are never
added. Pairwise binding does not impose a hard logical identity AND state gate.

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
- Grounding and retrieval train semantic projection/composition/context. Retrieval
  keeps pairwise softplus ranking and optimizes the complete NEW fine score.
  There is no additional person semantic auxiliary loss in this version.
- Global state keeps same-required-identity positive/negative supervision at image
  level. Wrong-identity images have unknown state labels and are excluded.

Within Top-M:

```text
S_final = z_topM(S_fine_new) + fine_coarse_weight * z_topM(S_coarse)
```

Weight zero uses raw fine scores (the same ordering). The remaining gallery keeps
coarse order. Periodic evaluation, checkpoint selection, sampling, mining, AMP,
LRU and CPU prefetch retain the existing protocol. `best.pt` maximizes
`0.5 * overall val Full-mAP + 0.5 * macro-case val Full-mAP`.

## Architecture controls

`model.representation: shared` is a retrained old-style control: one identity
head feeds identity loss/coarse, instruction composition and the target-token
identity addition. It has no explicit pairwise binding and requires
`model.binding_mode: none`. `dual` has independent raw-feature heads and detached
identity binding. The control is not backward compatibility for old checkpoints.

DINO/FAFA x shared/dual tests backbone and the complete representation/routing
change. Binding removals compare `both`, `identity`, `semantic`, `none`; they
remove score terms, not semantic context or coarse identity. Evaluate by case and
CandidateRecall@500 as well as ID-mAP/Full-mAP. See [ablations](ablations_vi.md).
