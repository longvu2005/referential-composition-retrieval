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

A trainable identity head produces normalized identity embeddings:

```text
v_i = Normalize(P_id(h_i))
```

The expensive image encoder is frozen behind the feature cache. `P_id` remains
inside `RCRModel` and is trained normally; identity embeddings are therefore
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
trainable text/image projections map both to `model.state_dim`, followed by L2
normalization. State is a rough global action/context signal; precise identity
binding is handled by the fine reasoner. The default method YAML standardizes
the two scalar scores per query, over their common finite split-gallery support,
excluding self. Population standard deviation is computed after all scoring
chunks have been gathered. A constant branch contributes zero. This normalization
uses no relevance labels and adds no learned parameters.

`retrieval.coarse_normalization: none` uses raw `S_id + beta * S_state` instead.
`retrieval.coarse_beta` overrides the checkpoint beta; null inherits it. The
fallback value 0.4 is a starting point, not a measured optimum. Tune beta on val
with `configs/ablations/coarse.yaml`. The same numeric beta has a different
relative effect after normalization. `coarse_mode` selects `identity_only`,
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

The final Top-M order is determined only by `S_f`; the coarse score is used only
for shortlisting. Images outside Top-M retain their coarse order so the saved
output remains a complete split-gallery ranking. No coarse, identity, or coverage
score is manually added to the fine score.

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
  grounding supervision instead of treating every detection as a negative;
- `L_id`: supervised contrastive loss (default temperature `0.1`) over query
  persons and persons from valid positive target images, using one shared identity
  label vocabulary. Unknown identity `-1`, padding, negative target images and
  self-pairs are excluded. Repeated copies of the same cached image/person crop
  count once per batch; different images of the same identity remain positives;
- `L_ret`: mean pairwise `softplus(S_f(q,n) - S_f(q,p))` over valid
  positive/negative pairs;
- `L_state`: the same masked pairwise ranking loss applied to
  `S_state / state_temperature` (default `0.1`), using the sampled Full Positive
  and negatives. It trains both state projections and the shared text encoder;
  there is no in-batch negative assumption. `state_weight` defaults to `1.0`.
  Fine ranking uses only `S_f`; neither state nor identity is added to its score.

Candidate sampling picks one reviewed positive and excludes **all** other
known positives and the query image from that query's negatives.
Including positive targets supplies cross-image identity pairs even when query
identities do not repeat within a batch. `L_id` can still be zero when detected
persons lack known matching identities; target images are not assigned the query's
labels by assumption. GT-aligned cache labels determine identity matches.
