# Isolating a jointly learned person representation

This experiment changes only the person representation. `dual` remains the
default and retains its existing state-dict keys, computation and gradient
isolation. The separate legacy `shared` control is not used here because it
changes binding and adds identity to target context tokens.

## Representation and gradients

With the frozen cached FAFA feature \(h_i\), `shared_person` learns:

\[
u_i=\operatorname{LayerNorm}(W_u h_i+b_u),\qquad
z_i^{id}=\operatorname{norm}(W_{id}u_i+b_{id}).
\]

Query and target use the same `person_proj` (`Linear + LayerNorm`). Its input
width is the person cache width, independent of the scene input width.
`identity_head` reads this projected `u`, not raw `h`. No independent semantic
projection is instantiated. The existing `encode_semantic` API returns `u` in
this mode. Grounding, `StructuredComposition` and aligned target evidence read
`u` directly; `binding_scores` normalizes target `u` only for cosine scoring.
FAFA/DINO feature extraction and frozen BERT are unchanged.

The loss definitions/weights remain:

\[
L=\lambda_gL_{ground}+\lambda_{id}L_{id}
  +\lambda_rL_{ret}+\lambda_{state}L_{global}.
\]

- Identity loss trains the person projection and identity head.
- Grounding BCE trains grounding and the person projection.
- Fine retrieval trains composition, semantic/context evidence and the person
  projection on batches with valid evidence. Both query and target paths retain
  gradients. Identity cosine and assignment remain detached, so retrieval does
  not directly train the identity head.

Membership remains detached and case-free, with background logit zero, the same
thresholds/masks/NMS and no top-1 fallback. Partial global one-to-one identity
matching, private nulls, strict identity threshold and stable ties are unchanged.
All marker mentions and complete change text still enter `StructuredComposition`.
Target context uses dual's routing, without the legacy identity-vector addition.
EvidenceBinding, geometry, dropout and AMP protection are unchanged.

All four binding modes retain the same assignment and null penalty. Scoring
remains \(S_{fine}=\operatorname{softplus}(\theta)B+C S_{ctx}\).
The coarse state/identity branches, normalization and Top-M fusion remain
unchanged. There is no residual gate, additional loss, amplitude change, sampler
change or inference GT identity/cardinality input.

## Run both variants with the same conditions

Configure `configs/proposed.yaml` with your existing FAFA person and DINO scene
cache paths. The default person backend is FAFA. No cache rebuild is needed.
The paired suite overrides only `model.representation`; it creates separate
checkpoints and outputs, and uses the same seed, 20-epoch schedule, sampling,
loss weights, validation selection rule and fixed retrieval coefficients.

```bash
.venv-proposed/bin/python tools/run.py ablate \
  --config configs/ablations/shared_person.yaml --splits val
```

Outputs are `runs/ablations/shared_person/{dual,shared_person}/`; each has
`best.pt`, `last.pt`, training diagnostics and `val/` rankings/metrics. The suite
also writes a comparison `summary.csv` and configs recording all settings.
Optional paired seeds: add `sweep: {train.seed: [0, 1, 2]}` to both experiments.

Equivalent individual commands, using existing caches:

```bash
.venv-proposed/bin/python tools/run.py run --config configs/proposed.yaml \
  --train --splits val \
  --set model.representation=dual output.dir=runs/representation-dual \
        wandb.name=representation-dual

.venv-proposed/bin/python tools/run.py run --config configs/proposed.yaml \
  --train --splits val \
  --set model.representation=shared_person \
        output.dir=runs/representation-shared-person \
        wandb.name=representation-shared-person
```

`run --train` sets each checkpoint path to that output directory's `best.pt`.
For subsequent standalone retrieval/evaluation use its saved `run_config.yaml`:

```bash
.venv-proposed/bin/python tools/run.py run \
  --config runs/representation-dual/run_config.yaml --splits val
.venv-proposed/bin/python tools/run.py run \
  --config runs/representation-shared-person/run_config.yaml --splits val
```

Use `--splits test` on those saved configs for a frozen test evaluation. The
paired comparison keeps the same existing 0.4/0.4 fusion weights; tune separately
only as a distinct experiment using the existing validation calibration workflow.
Inspect overall/by-case Full-mAP and CandidateRecall@500. No quality improvement
is established by structural tests; validation determines the result.

## Checkpoint contract

`dual` and legacy `shared` retain architecture tag
`person-id-sem-partial-match-v2`. Existing dual V2 checkpoints without a new
top-level representation field remain compatible. `shared_person` uses
`shared-person-partial-match-v1` and must retrain. New checkpoints record both
representation and architecture tag. Run metadata records the loaded model.

Retrieval rejects a config requesting a different representation before loading
data or backbones. It also rejects disagreement between checkpoint representation
and saved config, and incompatible architecture tags. State dicts load strictly.
Training validation, mining and standalone retrieval use the same RCRModel and
ranking implementation. Caches and evaluator contracts remain unchanged.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-proposed/bin/python -m pytest -q
.venv-proposed/bin/python -m ruff check src tools tests labelstudio
```

`tests/proposed/test_shared_person.py` checks isolated loss gradients, actual
optimizer updates, target autograd, direct `u` routing, detached membership and
assignment, null/zero-person cases, nonfinite padding, CPU/CUDA AMP where
available, exact checkpoint round trips and representation rejection.
The train CLI tests cover both modes, mining, periodic validation and exact
ranking equality after standalone checkpoint loading. Split-cache tests exercise
both modes with different FAFA-person/DINO-scene widths and read-only cache reuse.
CUDA tests skip without a CUDA device. Tiny local fixtures do not measure PIPA
retrieval quality or native FAFA/CUDA throughput.
