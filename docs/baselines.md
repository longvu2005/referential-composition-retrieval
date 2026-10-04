# CLIP and FAFA baselines

Each baseline uses the existing final dataset, split galleries and official
ID/Full evaluator. Raw PIPA images and pretrained weights are not checked in.
Run all commands from the repository root.

## Environments

Use separate environments for proposed, CLIP and FAFA. FAFA uses its pinned
upstream source and `transformers==4.39.3`; proposed uses Transformers 4.56+.
Do not install both extras in one environment or substitute a PyPI LAVIS fork.
Example for CLIP (replace `clip` with `fafa` for FAFA):

```bash
python3.11 -m venv .venv-clip
source .venv-clip/bin/activate
python -m pip install -r requirements/bootstrap.txt
python -m pip install --no-build-isolation -r requirements/clip.txt
```

The bootstrap pin retains `pkg_resources` needed by upstream CLIP. On Kaggle,
call `.venv-clip/bin/python` or `.venv-fafa/bin/python` directly. The runtime
never creates environments or installs packages.

## CLIP

```bash
python tools/methods/run.py run --config configs/methods/clip.yaml --prepare
# Reuse prepared weights/cache:
python tools/methods/run.py run --config configs/methods/clip.yaml
# Split the experiment into val selection and frozen test:
python tools/methods/run.py run --config configs/methods/clip.yaml --splits val
python tools/methods/run.py run --config configs/methods/clip.yaml --splits test
# Run only image/text baselines:
python tools/methods/run.py run --config configs/methods/clip.yaml --modes clip_image clip_text
```

One official OpenAI CLIP checkpoint contains both encoders. There is no separate
fusion model, detector, LLM or RCR training. Both native TorchScript and raw
OpenAI CLIP state dictionaries are supported.

| Mode | Scoring |
| --- | --- |
| `clip_image` | Query image · gallery image |
| `clip_text` | Query text · gallery image |
| `early_fusion` | Normalize weighted query image/text sum, then dot gallery |
| `late_fusion` | Weighted sum of branch scores after per-query gallery z-score |

Image/text features are normalized. Image features, text features and branch
score arrays are cached; fusion reuses these arrays. A warm cache does not load
the encoder. Cache fingerprints include checkpoint, precision, ordered image
paths/sizes/mtimes, and query text. Moving image files can invalidate the cache.

`run` selects early/late weights independently on **complete val**, with
`image_weight=alpha`, `text_weight=1-alpha`. Default grid: 0–1, step 0.05;
criterion: Full-mAP. Ties prefer alpha nearest 0.5, then smaller alpha.
Selection is saved before touching test inputs. Test-only requires matching val
provenance; changed weights, val labels, grid or text require a new val run.
Changing test labels cannot affect the selection. Val-optimal does not guarantee
test-optimal performance. Running val with only one fusion mode replaces the
selection file with that mode; select both if both will be tested later.

For a fixed-weight ablation or retrieval smoke check:

```bash
python tools/methods/run.py retrieve --config configs/methods/clip.yaml \
  --modes early_fusion --splits val --max-queries 2 \
  --set fusion.image_weight=0.4 fusion.text_weight=0.6
python tools/methods/run.py evaluate --config configs/methods/clip.yaml \
  --modes early_fusion --splits val --max-queries 2
```

Low-level `retrieve` uses YAML weights; it does not read `tuning.json`.
Use the same mode/split/subset when evaluating saved results.

## FAFA

```bash
python tools/methods/run.py run --config configs/methods/fafa.yaml --prepare
python tools/methods/run.py run --config configs/methods/fafa.yaml
```

`prepare` gets the pinned official source, released checkpoint, CLIP selector,
Faster R-CNN weights and FAFA runtime assets. Model code remains in the authors'
checkout. The configured commit, runtime cache and checkpoint checks remain;
`--force` refreshes prepared assets. Asset preparation can require substantial
RAM, disk space and network access. Retrieval uses the prepared assets.

FAFA is a single-person model adapted here to RCR scenes: detect people, select
query members with CLIP text/box similarity, encode each person/change component,
then aggregate native FDA scores with Hungarian SetMatch. Group membership is
predicted with threshold/margin and disjoint anchors, without GT identity or
cardinality. Group/relation conditions are approximated independently, not
jointly reasoned. These adapter settings are validation hyperparameters, not
measured optima, and are not tuned automatically by the CLIP fusion runner.

The long `fafa.py`/`fafa_adapter.py` files have separate responsibilities:
upstream model/runtime integration versus scene/Subject adaptation. Both are
needed for a faithful baseline and are retained.

## Saved results

`output.dir` is the run root (default `runs/clip` or `runs/fafa`). The CLI appends
mode for CLIP and then split. Each folder contains `scores.npy`, `rankings.pt`,
`run.json`, `metrics.json`. CLIP selections are in `<root>/tuning.json`;
`<root>/summary.csv` describes only the latest successful run invocation.

The evaluator requires complete split-gallery rankings with self excluded.
It reports Full/ID mAP and R@1/5/10 plus by-case/per-query metrics. Baselines do
not fabricate coarse candidate metrics. Test currently has only 14 queries;
use it as pipeline validation, not a publication-scale final benchmark.
