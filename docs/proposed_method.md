# Proposed method

The RCR model combines identity evidence with the requested visual change.
It uses a coarse retrieval stage followed by detailed scoring of shortlisted
images.

## Features

Grounding DINO provides person detections. Frozen CLIP image features represent
the scene. The pinned CLIP text tower supplies token and global features, with
rank-8 LoRA on the query and value projections of its last two blocks in the
supplied configuration. Frozen FAFA encodes person crops and text-conditioned
person features.
See the [feature contracts](feature_contracts.md) for compatibility requirements.

## Grounding and composition

Grounding associates Subject descriptions with people in the reference image.
`grounding.binding` conditions reference-image patches on each Subject
description. Each person token attends to those patches, and the grounding head
scores the resulting token. The grounding loss supervises these logits, and
the coarse identity loss uses their sigmoid values to weight query detections.

For each valid crop, a softmax across active Subjects and background sets its
confidence `1 - P(background)`. Selection requires confidence at least
`selection_threshold` and an active Subject with a raw logit strictly above
`background_logit`. Duplicate-box suppression uses the same confidence. The
winning Subject receives that crop's full confidence as detached membership.

`model.fine_matching.owner_mode` controls the choice among eligible Subjects.
`raw_logit` picks the highest raw score. `crop_log_softmax` uses

```text
owner_score(s, i) = logit(s, i) - logsumexp_j(logit(s, j))
```

The sum runs over all valid crops `j` in the same query image before selection
and duplicate-box suppression. The winner maximizes this score among active
Subjects whose raw logit for `i` exceeds `background_logit`. Ties use the
higher raw logit, then the existing Subject order. With a single valid crop,
both modes use the baseline owner. Each selected crop has one owner and each
Subject may contain several people. Composition combines these memberships
with Subject roles and the requested change.

Membership and identity matching use predicted evidence at inference. Ground
truth annotations provide training supervision and evaluation labels.

## Retrieval and reasoning

Coarse retrieval combines identity similarity with global image-text state
similarity. The fine stage reranks the coarse shortlist using a shared partial,
one-to-one identity assignment across Subjects.

The fine score combines native FAFA evidence, identity coverage and a single
context score read from CLS after joint reasoning. Person tokens keep their
Subject roles and individual GROUP members throughout the context path.

### Aligned person evidence

Let `R = [r_CLS, R_text, {r_ik}]` be the composed query, `m_ik` its detached
membership, and `pi(k)` the shared partial assignment (`-1` means no match).

```text
selected(i,k) = subject_active(i) AND (m_ik > 0)
matched(i,k)  = selected(i,k) AND (pi(k) >= 0)
g(box)       = [cx, cy, width, height]

o_ik = LN_target(W_e e_k + W_g g(box_pi(k)))  when matched(i,k)
     = n_miss                               when selected but unmatched
     = 0                                    otherwise

X = LN_inject(R + Dropout([0_CLS, 0_text, {o_ik}]))
```

`reasoning_binding` conditions target patches on the composed query and
produces `e_k` once per aligned reference detection. The model routes the
result into the Subject-person slots.

During the first `train.context_warmup_epochs` epochs, `grounding.binding` and
`reasoning_binding` refer to the same `EvidenceBinding` instance. Any validation
reasoning in this phase reads the warmed binding directly. At the start of the
next epoch, before mining or training, the model copies its parameters and
buffers into an independent `reasoning_binding`. Grounding retains its
optimizer moments and step counters. Reasoning begins with fresh optimizer
state and learns independently from context supervision. With zero warmup
epochs, this split happens before epoch 1.

`TargetPersonBuilder` projects evidence and center/size geometry and normalizes
their sum, following the aligned person design in `2221c4e`. `n_miss` is one
learned vector shared by missing members and empty endpoint sets. Injection
normalizes the entire reference sequence. Unselected slots are masked as keys;
a selected member remains available when its target match is missing.

### Ordered relation tokens

Before target injection, average the composed text tokens marked by each
Subject to obtain `a_i`, respecting `subject_token_mask` and `change_mask`.
When the marker mask is absent, use `subject_pos`. Cache one condition for each
slot pair `i < j` in `QueryEncoding`:

```text
u_ij = W_query concat(a_i, a_j)
c_ij = LN_condition(u_ij + MHA(u_ij, R_text, R_text; text_mask))
```

For each candidate, let `x_sk` be the injected person token in `X` and
`K_s = {k | selected(s,k)}`. Read the two endpoints separately:

```text
alpha_(s,k|ij) = softmax_over_K_s(dot(c_ij, x_sk) / sqrt(d))
v_(s|ij)      = sum_over_K_s(alpha_(s,k|ij) * x_sk)
p_ij          = W_pair concat(c_ij, v_(i|ij), v_(j|ij))
```

The shared `W_pair` always receives endpoint `i` before endpoint `j`. The member
set includes selected people without a match. An empty member set reads
`n_miss` directly, avoiding an empty softmax. A pair is a valid attention key
when both Subjects are active, even if an endpoint has no members or evidence.
A one-Subject query has zero pair tokens. Reconditioning rebuilds the same
representation from new text while retaining the original membership.

### Joint attention and CLS score

```text
Z0 = concat_tokens(X, {p_ij})
Z1 = LN_attn(Z0 + Dropout(SelfAttention(Z0; A)))
Z2 = LN_attn(Z1 + Dropout(SelfAttention(Z1; A)))
Z  = LN_out(Z2 + FFN(Z2))
h_cls = MLP(Z[CLS])
H(q,t) = where(matched.any_over_Subject_and_person, h_cls, 0).float()
```

Both attention rounds share parameters, normalization and bias `A`; the FFN
runs once afterward. Only CLS produces a scalar. `A` uses `query.prior` for
reference keys, zero for valid pair keys, and `-inf` for masked keys. Matched
person-to-person edges additionally receive a per-head directed geometry bias:

```text
Linear_geo([cx_b-cx_a, cy_b-cy_a, log(w_b/w_a), log(h_b/h_a), IoU(a,b)])
```

All other edges receive zero additional geometry. Sizes and IoU denominators
are clamped at `1e-6`; geometry and biased self-attention run in FP32 under AMP.
The match gate is equivalent to `coverage > 0` for this assignment. It uses
`where`, so all-missing targets return exactly zero. Partial coverage does not
scale the CLS score: `C=0.5, h_cls=0.8` gives `H=0.8`.

## Objective and ranking

The shared linear scorer combines native FAFA binding `B` (with the configured
pooling), matching coverage `C`, and raw context `H`:

```text
F = model.binding_scale * B + a_train * C + b_train * H

L_sem = retrieval_loss(b_train * H / tau_c, P, I & (C > 0), pair_mask)
L_ctx = L_sem + rho_effective * L_rej
```

`P` is Full Positives, `I` is identity positives, and `N` is hard identity
negatives. `retrieval_loss(S,P,V)` uses every positive/negative pair within `V`
and contributes `softplus(S_negative - S_positive)`. It averages pairs within
each query, then queries with at least one valid pair. Semantic context starts
with same-ID pairs with positive coverage.

With `loss.context_verify_members=true`, a predicted assignment verifies a
member if the reference and target crops have the same annotated ID and the
reference crop belongs to its predicted Subject. A pair must verify the same
set of distinct member IDs in both images for every active Subject, with at
least one member per Subject. This allows equal partial sets within a group.
The labels filter the semantic loss pairs after matching. The reasoner still
receives the predicted assignment. The control setting `false` uses the
original same-ID and coverage criteria.

`loss.rejection_loss_mode=softplus` retains the baseline rejection objective:

```text
L_rej = retrieval_loss(F / tau_c, P, (P | N) & (C > 0))
```

The supplied experiment uses `loss.rejection_loss_mode=hinge`. For each Full
Positive `p` and mined wrong-ID negative `n`, both with `C > 0`, it computes:

```text
D(t) = model.binding_scale * B(t) + a_train * C(t)
delta_0 = stopgrad(D(p) - D(n))
residual = b_train * (H(n) - H(p))
pair_loss = relu(residual - max(delta_0, 0)) / tau_c
```

For a positive base margin, context is penalized only when it makes the wrong-ID
image overtake the Full Positive. For a nonpositive base margin, context is
penalized only when it widens the wrong-ID lead; narrowing that lead has zero
loss even if the Full Positive still ranks below it. The whole base margin is
detached, so this term trains only the context contribution. The scorer's
`context_contribution` already includes `b_train`; temperature is applied once.

Both rejection modes use the same masks and reduction: average all valid pairs
within each query, including zero hinges, then average queries with at least one
valid pair, including queries with zero loss. Invalid values are masked before
score differences. A batch without valid pairs returns a differentiable zero.
`context_rejection_weight` multiplies the reduced rejection loss.
Membership, assignment, `B` and `C` are detached in the context loss.
After the split, gradients through `H` reach the reasoner, composition,
target-side binding and shared feature projections. Grounding and coarse
supervision continue to train the query-side binding.

The supplied configuration uses `context_weight=1.0`, `tau_c=0.1`, rejection
weight `0.5`, swap weight `0.0`, `a_train=0.04`, and `b_train=0.10`. With binding
scale `1`, `B=0.7`, `C=0.5`, `H=0.8`, the fine score is `0.80`, the semantic
logit is `0.8`, and the baseline softplus rejection logit is `8.0`.
Epochs 1–4 warm up the base and text LoRA with context disabled.
Epochs 5–6 enable semantic context. From epoch 7, mining and rejection start
with `rho_effective=0.5`; it is zero earlier. These settings are controlled by
[configs/proposed.yaml](../configs/proposed.yaml).

The loss API and supplied experiment default to no swap supervision
(`loss.context_swap_weight=0.0`). When enabled, swap supervision reconditions
text and calls the same aligned scorer using the original assignment for both
sentences. Both deferred
scene batches and materialized scene tensors are supported.

The hinge loss reuses the forward's `B`, `C` and weighted `H` in FP32 and uses
fixed-shape tensor operations with deferred numerical checks. It introduces no
device scalar reads, dynamic boolean indexing or extra scorer/cache calls.
The GPU input path keeps sparse eligible rows from `context_layout`, chunked
`SceneTargetBatch` scene loads and assignment-based `NativeTargetBatch.load_matched`
FAFA loads. Matching and cache I/O stay outside activation checkpointing;
backward recomputes the neural context path. The default prefetch depth remains
one batch; `train.prefetch_batches=0` runs the serial control. See
[GPU performance](gpu_performance.md) for profiling and runtime settings.

Base mining uses `b=0` on the screened semantic/wrong-ID union. For one rejection
slot, it also computes H on the screened wrong-ID group (including zero overlap
and partial identity). It reuses that group's B and C for the base union, while
semantic candidates skip the reasoner. On the same wrong-ID list, it compares
`z(B' + a*C) + eta*z(G)` with `z(B' + a*C + b_mine*H) + eta*z(G)`, where
`B'=model.binding_scale*B` and `b_mine=0.005`. Population z-scores precede the
`C>0` filter; ranks and the top-ten test follow it, with coarse order breaking
ties. The two highest H-ranked candidates with improved rank form a pool for
one random batch slot; the other hard slots use the base wrong-ID pool. See the
[run guide](proposed_runs.md) for quotas, partial-ID sampling and refresh timing.
The mining coefficient affects selection only. Validation collects raw `H`
before multiplying by `b`. The existing Top-M fusion scorer
calibrates `(a,b,eta)` and selects the checkpoint; the rest of the gallery keeps
its coarse order. Calibrated coefficients serve retrieval, while losses keep
`a_train,b_train` from the original training config. Test uses the validation
selection unchanged under the [benchmark protocol](benchmark_protocol.md).

## Configuration and checkpoints

Model and experiment settings live in YAML. Checkpoints record the effective
model configuration and feature provenance. Train `fafa-aligned-pair-cls-v5`
in a fresh run directory using compatible frozen feature caches.
Checkpoints record `binding_shared` to restore the binding phase during
retrieval and resume. Joint-phase resume restores both learned bindings and
their optimizer states. Warmup resume retains sharing until the split epoch.
For v4 inference, retrieval restores its shared binding.
After training v5, generate new retrieval artifacts to capture its learned `H`.
CPU replay uses the raw components saved with each ranking.
Use the [run guide](proposed_runs.md) for execution and the
[cache guide](detection_cache.md) for feature preparation.
