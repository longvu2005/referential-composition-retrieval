# Proposed method

The implemented inference path is:

```text
person detector -> shared image encoder -> Subject grounding
-> coarse identity + state shortlist -> structured identity-set composition
-> target evidence binding -> fine target-set reasoning -> ranking
```

### 1. Shared visual representation

A `person`-prompted detector supplies every detected person candidate; detected
heads do not gate candidates. A shared image encoder `E_I` produces:

- letterboxed whole-scene patch features `F`;
- full-person crop features `h_i`.

The cache retains raw 768-dimensional DINO features. One shared trainable
visual projection maps scene patches, person crops and global patch means to
384 dimensions. A separate trainable text projection maps frozen BERT tokens
from 768 to 384 dimensions. BERT parameters, including Subject-token embeddings,
are frozen; the backbone stays in eval mode during training.

A trainable identity head produces 128-dimensional normalized embeddings:

```text
F = P_visual(F_raw)
h_i = P_visual(h_i_raw)
v_i = Normalize(P_id(h_i))
```

The expensive image encoder is frozen behind the feature cache. `P_id` remains
inside `RCRModel`, alongside `P_visual`, and is trained normally; embeddings are
**not** stored permanently in the cache. Scene-box coordinates are transformed
to the same letterboxed coordinate system as the scene patch grid. GT head
annotations are used only to align optional identity supervision and compute
evaluation labels; no head crop enters the model.

### 2. Shared evidence binding

One `EvidenceBinding` module is shared by query grounding and target evidence
binding. Its order is **Scene → Reference → Person → Conditioned Scene**.

First, reference tokens condition the scene patches:

```text
F_bar = LN(F + Attn(F, R, R))
```

Then each person reads that conditioned scene using a soft geometry bias from
its box in letterboxed scene coordinates:

```text
e_i = Attn_geo(h_i, F_bar, F_bar; B_i)
b_i = FFN_res(h_i + e_i)
```

The geometry is a soft attention bias, not a hard crop mask, so relational and
contextual evidence may remain outside the person box.

All RCR attention modules use 6 heads at width 384. Binding, composition and
fine reasoning use FFNs with hidden width 768 (`mlp_ratio=2`). Attention weights,
attention residual updates and FFN hidden/output activations use dropout 0.1
during training; evaluation disables it.

### 3. Query Subject grounding

For Subject `s`, the shared text encoder `E_R` encodes its selection text. The
same encoder processes the change text; the shared Binding produces one raw
grounding logit per query person:

```text
g_si = Ground(F_q, h_i^q, B_i^q, T_sel^s)
m_si = sigmoid(g_si)
```

Membership uses independent sigmoid probabilities, never a softmax across
persons; a GROUP Subject can therefore select multiple people. Training uses
masked `BCEWithLogits` on raw finite logits.

### 4. Cheap coarse identity + global state retrieval

Coarse retrieval is deliberately optimistic and is used only to preserve high
candidate recall.

```text
c_i(q,t) = max_j dot(v_i^q, v_j^t)
a_si = sigmoid(g_si)
omega_si = a_si / (sum_r a_sr + eps)
S_subject_s(q,t) = sum_i omega_si c_i(q,t)
S_id(q,t) = average_s S_subject_s(q,t)
z_text(q) = Normalize(P_text(masked_mean(E_R(final_change))))
z_image(t) = Normalize(P_image(mean_patches(F_t)))
S_state(q,t) = dot(z_text(q), z_image(t))
Z_branch(q,t) = (S_branch(q,t) - mean_q) / max(std_q, 1e-6)
S_coarse(q,t) = Z_id(q,t) + beta * Z_state(q,t)
```

Each Subject contributes once to the average, including when one Subject
contains multiple people. Invalid person/Subject positions are masked. There
is no threshold, one-to-one assignment, or coverage heuristic in `S_id`; its
soft grounding + max identity similarity formula is unchanged. The state branch
uses only the existing `final_change` token encoding (including Subject markers),
with padding excluded from mean pooling, and whole-image patch means. Separate
trainable text/image projections map both to `model.state_dim=128`, followed by L2
normalization. State is a rough global action/context signal; precise identity
binding is handled by the fine reasoner. The default method YAML standardizes
the two scalar scores per query, over their common finite split-gallery support,
excluding self. Population standard deviation is computed after all scoring
chunks have been gathered. A constant branch contributes zero. This normalization
uses no relevance labels and adds no learned parameters.

`retrieval.coarse_normalization: none` uses raw `S_id + beta * S_state` instead.
`retrieval.coarse_beta` overrides the checkpoint beta; null inherits it.
The main YAML uses beta=0.4 during training-time validation;
the model-level fallback is also 0.4. After training,
`configs/ablations/fine_coarse.yaml` jointly sweeps beta and the final
fine/coarse weight on val, selecting the pair by final Full-mAP at fixed Top-500.
CandidateRecall@500 is diagnostic. `coarse.yaml` uses the selected pair only for
optional mode comparisons. The same numeric beta has a different relative
effect after normalization. `coarse_mode` selects `identity_only`,
`state_only`, or `identity_state`; beta=0 in the combined mode also disables state.
Top `M` gallery images proceed to fine when `retrieval.rerank: true`.

### 5. Structured identity composition

The change text contains distinct `[S1]` / `[S2]` tokenizer special tokens
aligned with Subject IDs. Each separate identity token uses the reference-free
query identity `v_i` plus the embedding for its Subject role.

Independent grounding probabilities are normalized only when a Subject marker
needs a Subject-level identity summary; the individual identity tokens remain
available to the fine reasoner. Identity-token keys receive the additive
`log(sigmoid(g_si))` membership prior in composition attention, target scene
binding, and fine self-attention. CLS/change keys have neutral prior; padding
is masked. No hard membership threshold removes a valid identity token.

This gives:

- permutation invariance among identities inside one Subject;
- explicit role distinction between Subjects; and
- individual identity tokens for final target matching.

### 6. Target evidence binding and fine reasoning

For each coarse Top-M target, the same `EvidenceBinding` conditions its scene
with the **composed reference** (change, Subject roles, individual identity
tokens, and aligned membership prior). Target person tokens combine:

```text
LN(projected bound evidence + projected target person identity
   + projected scene-box geometry)
```

The composed query cross-attends to the whole target person set, then a
self-attention block scores its CLS token as the raw scalar `S_f(q,t)`.

Within Top-M, the default method YAML uses:

```text
S_final(q,t) = z_topM(S_f(q,t)) + 0.4 * z_topM(S_coarse(q,t))
```

Both population z-scores use the same shortlisted candidates with finite coarse
scores, after self exclusion; constant branches contribute zero. Unsupported
coarse candidates remain last. `retrieval.fine_coarse_weight` controls the
coarse contribution; zero returns raw fine-only scores. Images outside Top-M
retain their coarse order, preserving a complete split-gallery ranking.
The joint grid includes the default (beta=0.4, weight=0.4) and zero values for
both axes. Both chosen coefficients are saved and applied unchanged to test;
equal validation Full-mAP retains the first pair in YAML order. Training still
optimizes raw fine scores, not this inference-time fusion.

When the query has no detected people, it remains in retrieval and evaluation:
`S_id` is zero for nonempty target person sets, so global state can still
rank them. Identity-based modes retain the original `-inf` coarse score for empty
target person sets. `state_only` scores all targets independently of that mask.
The fine stage can still use change text
and target evidence. Empty targets and padded persons/Subjects do not crash.

### Training objective

The current objective is:

```text
L = 1.0 * L_ret + 1.0 * L_ground + 0.1 * L_id + 1.0 * L_state
```

where:

- `L_ground`: masked BCE-with-logits for independent Subject/person labels;
  a Subject whose annotated identities have no detected match is excluded from
  grounding supervision instead of treating every detection as a negative.
  Unmatched detections with unknown identity are also excluded from this loss;
- `L_id`: supervised contrastive loss (default temperature `0.1`) over query
  persons and persons from valid positive target images, using one shared identity
  label vocabulary. Unknown identity `-1`, padding, negative target images and
  self-pairs are excluded. Repeated copies of the same cached image/person crop
  count once per batch; different images of the same identity remain positives;
- `L_ret`: mean pairwise `softplus(S_f(q,n) - S_f(q,p))` over valid
  positive/negative pairs;
- `L_state`: the same masked pairwise ranking loss applied to
  `S_state / state_temperature` (default `0.1`). Only sampled images containing
  **all required identities** are eligible: Full Positives versus same-identity
  negatives. Eligibility comes from train image-level GT, not detected identity
  labels, so detector misses do not change state supervision. Wrong-identity
  images have unknown state labels and are excluded. Queries without an eligible
  positive-negative pair contribute no state loss. It trains both state
  projections, visual projection and text projection; BERT stays frozen. There
  is no in-batch negative assumption. `state_weight` defaults to `1.0`.
  `L_ret` uses raw `S_f`; inference uses the configured fine/coarse fusion.

Each epoch draws `N_train` samples with replacement. Per-sample weight is
`1 / sqrt(N_case)`, so the case probability is proportional to `sqrt(N_case)`.
Within a case, samples are uniform. Draws are then batched by Subject count,
without discarding any draws. `train.case_balanced=false` uses a shuffled pass
without replacement for the natural-distribution ablation.

Candidate sampling picks up to `P=2` distinct reviewed positives and excludes
**all** known positives and the query image from that query's negatives. At
least one candidate remains negative. For equivalent train instructions
(same ordered identities, case and exact change text), an image
marked positive by one query is not used as a negative by another. Original
positive labels are retained, never merged or rewritten; val/test are untouched.
With `C=24`, `P=2` and mining disabled, sampling draws 11 same-identity and
11 random negatives. The main YAML enables mining (`hard_fraction=0.3`):
11 identity, 6 mined and 5 random negatives after warmup. Quotas are floored
fractions of the actual `C-P` negative slots. A query with only one positive
uses 23 negative slots, giving 11 identity, 6 mined and 6 random negatives;
missing slots fall back to random. Candidates are unique. Mining uses only the
train split, shared coarse scoring in eval/no-grad mode and a bounded top-ranked
pool; it does not retain a train-query by full-gallery ranking matrix.

Including positive targets supplies cross-image identity pairs even when query
identities do not repeat within a batch. `L_id` can still be zero when detected
persons lack known matching identities; target images are not assigned the query's
labels by assumption. GT-aligned cache labels determine identity matches.

Loss ablations hold retrieval at identity-only for every variant, isolating
auxiliary supervision. Inference ablations compare identity-only and combined
shortlists using the same fully trained checkpoint. State projections from runs
with `state_weight=0`, or new checkpoints with zero supervised state pairs, cannot
be used for state-based retrieval. Existing checkpoints/cache formats remain
readable; benefiting from the new supervision requires retraining.

Periodic validation selects `best.pt` using:

```text
macro_full_map = mean(Full-mAP for each nonempty validation case)
checkpoint_score = 0.5 * overall_full_map + 0.5 * macro_full_map
```

The four cases receive equal weight in the macro term when all are present.
Selection uses the final complete-gallery ranking, including configured
fine/coarse fusion. Equal scores keep the earlier checkpoint. History, W&B and
checkpoints record the selection score and both mAP components; test is never
used for epoch selection.
