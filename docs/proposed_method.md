# RCR soft partial matching (v2)

Architecture ID: `rcr-soft-partial-v2`. Semantic cache ID: `clip-person-text-v4`.
Old BERT/shared-semantic/state-fusion checkpoints are incompatible and must be
retrained. Loading is strict: no missing-key or partial checkpoint fallback.

## Inputs and ownership

Inference takes only query image features, selection text, the complete change,
and candidate image features. `parse_subjects` derives ordered Subject IDs,
counts and all mentions from text. Subject IDs up to `model.max_subjects` are
supported; larger IDs raise an explicit error. A Subject may contain any number
of detected members. The model never receives cases, identity annotations,
GT boxes, or cardinalities. Cases are used by the unchanged evaluator only.

All backbones are frozen and remain in evaluation mode. Training loads their
cached outputs and therefore does not keep any backbone on the accelerator.

| Tensor | Shape | Meaning / mask |
| --- | --- | --- |
| `persons` | B × K × Df | Mean FAFA image-only Q-Former hidden tokens, before vision projection |
| `clip_pooled` | B × K × Dc | Normalized CLIP projected pooled CLS; `person_mask` is B × K |
| `clip_tokens` | B × K × N × Dv | CLIP vision last hidden states; `clip_token_mask` is B × K × N |
| `scene` | B × P × Dd | DINO patches without CLS/register tokens |
| `boxes` | B × K × 4 | Detector xyxy boxes on the normalized letterboxed canvas |
| `selections` | B × S × Ld × Dt | Frozen CLIP selection hidden tokens; `selection_mask` |
| `change` | B × Lc × Dt | Full reassembled condition tokens; `change_mask` |
| `subject_token_mask` | B × S × Lc | Every token overlapping every mention of each role |
| `subject_ids`, `subject_mask` | B × S | IDs from text; padded roles are inactive |
| `membership` A | B × S × Kq | Soft categorical foreground memberships |
| `transport` P | B × Kq × (1 + Kt) | Column zero is null |
| `members`, `evidence` | B × S × Kq × D | Distinct role/member outputs, retained until reasoning |

Training adds a candidate axis C, then scores pairs in chunks. Batch dictionaries
have separate `inputs` and `supervision` branches; only the former enters the model.

## Frozen features and text

CLIP image and text use the same pinned `openai/clip-vit-base-patch32` revision
and its image processor/tokenizer. Implementation follows Hugging Face
`CLIPModel`: `vision_model(...).pooler_output` is post-layernorm pooled CLS;
`visual_projection` produces the pooled embedding. `last_hidden_state` contains
raw visual hidden tokens before the pooled post-layernorm. Text
`last_hidden_state` already has the text final layernorm. Only projected pooled
CLS/EOS embeddings were trained for CLIP contrastive alignment; intermediate
visual/text tokens are not claimed to be independently aligned. Trainable task
projections consume those hidden tokens.

The fast tokenizer supplies offsets on the original unchanged text. Token IDs
are split into non-overlapping windows of context length minus two, with native
BOS/EOS added to each window. We remove those boundary tokens after encoding,
concatenate all original content tokens in order, and use their original offsets
for all Subject mentions. Composition adds global sinusoidal positions to the
reassembled text, plus learned role embeddings at every mention. No text is
silently truncated and no condition is split into independent per-Subject tasks.
Pretrained context does not cross window boundaries; the trainable composition
Transformer reads across them. Very long texts still increase Transformer memory.

DINO/FAFA source caches remain immutable. CLIP adds only missing features. Source
IDs, source index checksums, gallery order, encoder/preprocessing metadata,
per-image detector boxes, dataset image registry, exact text set and sidecar
checksums are validated. Text changes refresh text features; annotations only
refresh the separate supervision sidecar. Cache IDs are retained across these
refreshes, but checkpoint text/supervision signatures distinguish their contents.
Legacy DINO metadata requires an explicit user assertion via
`cache.allow_legacy_dino=true`; it cannot prove the missing provenance. Default
configuration rejects that legacy format. Reusing raw source caches assumes their
image bytes and feature shards remain immutable; metadata is not a reconstruction
of an undocumented historical encoder run.

## Grounding and composition

Grounding owns its own appearance, text, scene and geometry projections. Query
person tokens read selection tokens and scene patches; local ROI scene features
and box geometry provide spatial context. For each person, a background logit
and one logit per active textual Subject form a categorical softmax:

\[
a_{si}=\operatorname{softmax}_{s=0,\ldots,S}(g_{si}),\qquad
\sum_{s=1}^S a_{si}\leq1.
\]

Padding is zero membership, not a background example. Multiple persons can have
high membership in the same Subject. There is no top-1 person selection.

Composition creates

\[
r_{si}=P_vv_i+e_s+P_b\operatorname{geom}(b_i)
\]

and reads `[sentinel, entire condition, all Subject/person tokens]` with a small
Transformer and key prior `log(a + 1e-8)` on member tokens. Roles are added at
all matching text spans. Member tokens have no arbitrary index positional
embedding, making the set permutation-equivariant. Text tokens retain order.

## Identity and differentiable transport

A shared identity head gives normalized FAFA projections. Its confidence is

\[
D_{ij}=z_i^{q\top}z_j^t,\qquad
p^{id}_{ij}=\sigma((\operatorname{softplus}(a)+10^{-6})D_{ij}+\beta).
\]

Row relevance is \(w_i=\sum_sa_{si}\). For real targets use
\(\ell_{ij}=\log p^{id}_{ij}\); null uses learned \(b_\varnothing\). We maximize

\[
\sum_{ij}X_{ij}\ell_{ij}-\tau_A\sum_{ij}X_{ij}\log X_{ij}
\]

subject to nonnegative X, row sums w, real-column sums at most one and unrestricted
null capacity. This is soft capacity-constrained matching, not hard one-to-one
assignment, independent row softmax or balanced Sinkhorn.

The FP32 log-domain solver minimizes the convex dual by alternating exact row
and column updates. With \(L=\ell/\tau_A\), row log scalings u and column log
scalings v:

\[
u_i=\log w_i-\operatorname{LSE}_j(L_{ij}+v_j),\qquad
v_j=\min(0,-\operatorname{LSE}_i(L_{ij}+u_i)),\quad v_0=0.
\]

A final row update restores the row equalities. Stop only when the row residual,
real-column violation and dual update residual all meet tolerance. Otherwise
raise an error; never return a violating solution as converged. Updates remain
in the autograd graph. Invalid rows/columns use finite masked log values to avoid
undefined all-masked logsumexp derivatives, then are explicitly zeroed. Empty
targets send all relevant mass to null. \(P=X/(w+10^{-8})\); zero rows give zero P.

## Target binding and joint condition reasoning

Each target person supplies raw CLIP visual tokens, 2 × 2 bilinear samples from
an expanded DINO ROI, and a geometry token. Coordinates use the existing
letterbox transform, normalized patch centers and `align_corners=False`.

Binding uses hierarchical attention: each reference member attends within each
person's evidence, then mixes those summaries with its transport P and a learned
null value. This preserves the person-level prior exactly; adding tokens to a
person does not multiply that person's matching mass. The epsilon residual and
zero-relevance rows are treated as missing/null, never redistributed across real
people. No CLS/readout attends directly to unbound target evidence.

Two small self-attention blocks jointly read the full composed text, reference
members, bound evidence, role markers, memberships, matching entropy and null
mass. An explicit missing marker exists for each empty Subject. Role ordering is
preserved; permuting member/target enumeration leaves the scalar score unchanged.
The joint logit is the principal condition decision, not a bounded correction.

## Retrieval score and shortlist

\[
q_i^{id}=\sum_{j\geq1}P_{ij}p_{ij}^{id},\qquad
S_{id}=\frac1S\sum_s
\frac{\sum_i a_{si}\log\operatorname{clamp}(q_i^{id},\epsilon,1)}
{\sum_i a_{si}+\epsilon},\qquad
S=S_{id}+\operatorname{logsigmoid}(z_{cond}).
\]

Here epsilon is 1e-8. An active Subject with membership mass at most epsilon has
identity score `log(epsilon)` and an explicit missing marker. If no Subjects are
active, the same finite low identity score applies. Null does not contribute
identity confidence. Scores are ranking confidences, not calibrated probabilities.

Coarse replaces \(q_i^{id}\) by the maximum valid real-target identity confidence;
it never calls transport or joint reasoning. An empty target has confidence
epsilon. Gallery projections are refreshed from the current identity head once
per retrieval/mining invocation, then reused in CPU chunks.

Default `retrieval.mode=shortlist` fine-scores top-M (initially 500), followed by
the unchanged coarse tail. No state branch, z-score, or coarse/fine weighted sum
exists. `full` fine-scores the complete self-excluded gallery; `coarse` is an
explicit identity-only diagnostic. All modes pass complete rankings to the same
unchanged evaluator, with identical positives and AP denominators. CandidateRecall
and CandidateHit come from the coarse shortlist. Runtime metadata records elapsed
retrieval time including feature I/O and gallery projection, seconds/query and
peak CUDA allocated bytes when on CUDA. It excludes initial data/checkpoint loads.
No empirical speedup or accuracy improvement is implied by pair-count reduction.

## Supervision and gradient routes

| Loss | Supervision | Trainable ownership |
| --- | --- | --- |
| Grounding CE | Known referenced identity → textual role; known unreferenced identity → background; unknown/conflicting → ignore | Grounding only |
| Identity SupCon | Shared train-only vocabulary; crop deduplication across query/candidate appearances; unknowns ignored; anchors without positives skipped | Identity head only |
| Matching NLL | Negative log sum of P over all usable same-ID detections | Matching calibration/null parameters |
| Multi-positive rank | Mean positive log-softmax over valid candidate scores / temperature | Matching, composition, target binding, joint reasoner |

Memberships and both identity embeddings are detached at retrieval/matching
boundaries. Grounding and composition do not share a trainable projection.
Backbone features are detached as well. Transport output itself is not detached.

GT heads only label existing detector boxes. A crop containing multiple annotated
head centers is conservatively unknown. Correspondence positives include all
usable detections of the same identity regardless of image-level condition.
A null positive is allowed only when all target detections have known labels and
either the GT identity is present with no usable detection (confirmed detector
miss), or the image explicitly declares `identities_complete=true`. Missing or
unknown annotations never imply absence. The current dataset has no completeness
flag, so absent identities do not automatically supervise null.

Warmup trains grounding and identity only. Main epochs keep those losses and
add matching/ranking. After a main epoch, hard mining uses the current coarse
pool and current fine scorer, on train queries/gallery only. Sampling reserves
a wrong-ID negative when available, includes same-ID/wrong-condition negatives,
and fills remaining slots with mined/random negatives. Self, every known positive
and disputed negative pairs are excluded. No case-balanced sampling exists.
Only overall validation Full-mAP selects `best.pt`; warmup checkpoints are never
selected as primary retrieval models.

See [run commands and verification boundaries](proposed_runs.md).
