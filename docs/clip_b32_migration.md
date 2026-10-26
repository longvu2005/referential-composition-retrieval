# CLIP ViT-B/32 backbone migration

This replaces Proposed's DINOv3 scene encoder and BERT text encoder with the
frozen image/text pair from `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The architecture is
`clip-b32-fafa-partial-match-v2`. Grounding DINO Tiny detection, FAFA person
features/native FDA, identity assignment, labels and evaluation are retained.

## Feature contract

- Scene input is a complete image, letterboxed to 224 x 224 with bicubic resize,
  then CLIP rescaling/normalization. No center crop discards people at the edges.
  Detector pixel boxes are mapped with that same letterbox geometry.
- The scene cache contains the 7 x 7 patch grid, width 768, separately from the
  native projected CLS vector, width 512. Global state never uses a patch mean.
- Native CLIP text projection supplies 512-D token states and the EOT vector.
  Context learns a separate 512-to-384 projection. The image context projection
  maps 768 to 384. Frozen native projections are never initialized randomly.
- State image/text adapters are 512-to-512 affine maps initialized to identity
  and zero bias. Their initial score is native CLIP cosine; the existing state
  loss then adapts them using only images with all required identities.
- CLIP keeps its original vocabulary and 77-position context, including BOS/EOS.
  Character offsets locate every complete `Subject 1` / `Subject 2` mention for
  the existing role injection. No randomly initialized special tokens are added.
  Truncation preserves EOS and is reported. Losing all complete mentions of a
  required Subject raises an error rather than silently changing the query.
- The separate native FAFA text policy remains 128 tokens. CLIP's 77 positions
  must not be described as a 128-token encoder.

The committed dataset was checked with the pinned real tokenizer: all 4,311
change texts and 4,800 Subject-selection texts fit within 77 tokens, with no
missing Subject spans. This statement applies to this export, not future edits.
Training saves `clip_text_stats` in the effective configuration; retrieval saves
query-scoped statistics in both `rankings.pt` and `run.json`.

## Build and run

Use the repository root, Python 3.11/3.12, and set `data.image_root` in
`configs/proposed.yaml` to the existing raw-image directory. Install/update the
two separate environments as needed. Proposed requires PyTorch >=2.6 because
the pinned CLIP release uses `pytorch_model.bin`; the Transformers loader does
not support this checkpoint with older Torch. FAFA keeps its own Torch 2.5.1:

```bash
PYTHON=python3 bash scripts/setup.bash proposed
PYTHON=python3 bash scripts/setup.bash fafa
```

The complete validation workflow is:

```bash
bash scripts/methods/proposed.bash val
```

Or run the stages explicitly:

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage detections
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage scene
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-image
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py calibrate --config configs/calibration.yaml
```

Outputs use `runs/proposed-clip-b32` and `runs/clip-b32-calibration`. Calibration
reads the saved training config, so its paths and training overrides are retained. Run test only
with the newly selected configuration:

```bash
bash scripts/methods/proposed.bash test
```

## Cache and checkpoint compatibility

The detection key is `data.detection_cache` (stage `detections`); scene extraction
uses `data.scene_cache` (stage `scene`) without running detection. Defaults are
`cache/grounding-dino-tiny`, `cache/proposed-clip-b32`, `cache/fafa-person-images`
and `cache/fafa-person-queries`. The FAFA baseline reads the same independent
detection cache. Detector, selector and crop order are unchanged. FAFA features
bind detection identity and no longer depend on the scene encoder.

Build new caches and train new learned modules. Old DINO/BERT weights cannot
resume this architecture, even where tensor widths happen to match. Cache
validation checks the image/text checkpoint, revision, detector, source cache
identity and separate scene/global dimensions. Completed unified CLIP scene
features can be reused and their boxes imported; format-7 FAFA caches must be
built once in the new paths. See [detection migration](detection_cache.md).
Old outputs remain available in
their original directories, and historical score artifacts retain CPU replay.

## Validation

The CPU suite covers training, standalone retrieval, calibration, full-gallery
export/replay and native CLIP contracts with tiny real Transformers models.
The released pretrained checkpoint was also loaded on CPU: scene output is
`[1,49,768]`, native global output is `[1,512]`, and wrapper image/text embeddings
and initial state cosine match the native Transformers outputs.

These checks validate the implementation. A full GPU training/retrieval run and
new benchmark mAP are separate experiments requiring the raw images.
