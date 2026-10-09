# OpenAI CLIP baselines

## 1. Architecture and checkpoint

Use pretrained OpenAI CLIP `ViT-L/14`, containing the image/text transformers
and contrastively learned projections. The implementation is pinned to
`a1d071733d7111c9c014f024669f959182114e33` in `pyproject.toml`.
`prepare` verifies the official download's SHA256 and saves
`checkpoints/clip/ViT-L-14.pt`; record its actual hash per run. Native state
dictionaries and official JIT archives are supported.
Sources: [official CLIP](https://github.com/openai/CLIP/tree/a1d071733d7111c9c014f024669f959182114e33),
[loader/preprocessing](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py).

## 2. Input, preprocessing and features

Encode whole RGB scene images with pinned OpenAI preprocessing: bicubic resize,
center crop to model resolution (224 for ViT-L/14) and CLIP channel normalization.
Text is `final_instruction` by default; `final_change` is an explicit alternative
experiment. Tokenization uses a 77-token context, `truncate=True`, retaining EOS.
Count overflowing queries using non-truncating tokenization and save
`truncated_text_queries`, including warm-cache runs. Text is not rewritten.

Features are L2 normalized. Eval/inference mode uses upstream fp16 weights on
CUDA, fp32 on CPU. Cache image/text features and branch scores using checkpoint
hash, precision, ordered image paths/sizes/mtimes and exact texts. Fusion reuses
cached branch scores; a warm complete cache avoids loading the encoder.

## 3. Retrieval equations

For normalized image/text embeddings x_q, t_q, x_g and image weight alpha:

\[
s_{image}=x_q^\top x_g,\quad s_{text}=t_q^\top x_g,
\]
\[
s_{early}=\operatorname{norm}(\alpha x_q+(1-\alpha)t_q)^\top x_g,
\]
\[
s_{late}=\alpha Z_q(s_{image})+(1-\alpha)Z_q(s_{text}).
\]

Z_q uses mean/population std over the **complete split gallery excluding q**,
std clamped to epsilon=1e-6. Constant branches contribute zero. Save a finite
self-score placeholder and exclude it from ranking. Early-mixture norm is
clamped to 1e-12. Batched scoring and stable sorting produce complete rankings.

## 4. RCR adaptation

Keep `clip_image`, `clip_text`, `early_fusion`, `late_fusion`. Adapt only the RCR
loader, instruction text and evaluator. No detector, GT box, identity lookup or
explicit person binding is added.

## 5. Hyperparameters and validation

`configs/clip.yaml` selects model/text field and runtime. Choose each fusion's
alpha independently from 0..1, step 0.05, by complete-val **Full-mAP**. Exact ties
prefer alpha closest to 0.5, then smaller alpha. `tuning.json` saves all trials,
chosen weights and validation-only context. Test rejects changed checkpoint,
text, val labels, grid or normalization and reuses the chosen weights.
Select both fusion modes together when needed; a new val selection contains the
requested modes and preserves the prior file.

## 6. Training policy

No training/fine-tuning on RCR. Only scalar weights are selected on validation.
Test labels cannot affect selection or its provenance.

## 7. Reproduction

Install the CLIP profile in this notebook as described in [baselines](../baselines.md), then run from the repo root:

```bash
python tools/run.py prepare --config configs/clip.yaml
python tools/run.py run --config configs/clip.yaml --splits val
# Once the test protocol is eligible and settings are frozen:
python tools/run.py run --config configs/clip.yaml --splits test
```

Outputs: `runs/clip/<mode>/<split>/{scores.npy,rankings.pt,run.json,metrics.json}`.
Low-level `retrieve` uses YAML weights, not `tuning.json`; use `run` for paper
fusion results. `--max-queries` results are smoke checks, rejected in reports.

## 8. Limitations

Whole-scene features mix identities/background/conditions; group membership and
ordered relations are not modeled explicitly. Long descriptions can truncate
the requested change; report the count. Early fusion and an unnormalized weighted
score sum induce the same per-query ranking, except a degenerate zero mixture.
Late fusion adds score normalization. CPU/CUDA precision may alter near ties;
keep precision/hardware fixed between val and test.
