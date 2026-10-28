# Independent person detection cache

Grounding DINO Tiny boxes are shared by Proposed, the standalone FAFA baseline,
train-only GT head alignment, and detection diagnostics. Scene extraction never
loads the detector; FAFA extraction never loads scene features.

| Config key | Default | Owns |
| --- | --- | --- |
| `data.detection_cache` | `cache/detections` | Ordered pixel boxes and original image sizes |
| `data.scene_cache` | `cache/scene` | CLIP patches and native global vectors |
| `data.fafa_image_cache` | `cache/person-images` | Packed person means and projected target tokens |
| `data.fafa_query_cache` | `cache/person-queries` | Content-addressed multimodal query features |
| `cache.supervision_cache` | `cache/train-supervision.json` | Train-only head alignment; separate from visual evidence |

All four feature-cache roots must be distinct, including after resolving symlinks and parent
paths. Set `data.image_root` to the actual image directory before extraction.
Commands below run from the repository root after installing the environments.

Cache paths name their role rather than a particular encoder or experiment.
To reuse completed caches after a directory rename, move each entire directory
with its index and shards intact and update the YAML paths used by that run.
Keep checkpoint bytes and stored cache IDs unchanged. Rebuilding is not needed
solely because a directory or the supervision JSON has a new name.

The FAFA baseline's validation lock includes its detector path. After changing
that path in `configs/fafa.yaml`, rerun FAFA validation before test to refresh
`runs/fafa/protocol.json`; the detector cache itself remains reusable.

## Build stages independently

```bash
# Grounding DINO Tiny only; requires the Proposed environment, not FAFA.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage detections

# CLIP only; no detection cache or FAFA environment is required for this stage.
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage scene

# Prepare FAFA assets, then extract from the shared pixel boxes.
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-image
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml --cache-stage fafa-query
```

The first two commands can run in either order. FAFA needs completed detections;
it does not need a scene cache. Running `build-cache` without a stage builds all
missing stages. Re-run an interrupted stage with the same settings to resume.
Completed caches can be mounted read-only for training and retrieval.

Changing scene extraction preserves detection and both FAFA caches. Changing
query text invalidates only affected FAFA query conditions. Changing accepted
boxes, their order, detector settings, FAFA weights or crop preprocessing
invalidates the dependent FAFA features. No second NMS, rounding or crop cap is
introduced. Detection stores no confidence scores; the baseline's score 1.0
remains an eligibility sentinel. Its existing zero-person fallback is retained.
Boxes that cross the image boundary retain their original coordinates; FAFA's
PIL crop padding and the scene transform's clipping preserve crop semantics.

## Supported detection inputs

Detection extraction, supervision and diagnostics use the independent pixel-box
cache only. Existing `person-detections-v1` indexes and native unfinished builds
remain reusable. Scene-to-detection migration is no longer part of the runtime
API; convert historical unified caches with their original code revision or
build a fresh detection cache before using this pipeline.

A completed unified CLIP scene cache remains usable as `data.scene_cache`; its
geometry is ignored by the scene reader. Legacy DINO scene features cannot
replace CLIP features. Old FAFA formats 4–6 must be rebuilt once for the separated
crop contract.

## Shared consumers and provenance

The detection index contains pixel boxes, image sizes, packed offsets, ordered
image IDs, the fixed `person` prompt and detector configuration. Its content ID
covers all of these fields and survives relocation. Readers verify that ID;
training labels and image/text features are absent. Image IDs refer to immutable
benchmark images; changing source pixels requires fresh affected caches.

FAFA format 7 binds this detection ID, native model/source/preprocessing,
checkpoint hash and extraction precision. It stores no scene dimensions, CLIP
checkpoint, or scene UUID. At training time `GalleryCache` joins the FAFA index,
detections and scene patches by canonical image order. Pixel boxes are mapped to
the scene's letterbox coordinates only at this join. The native CLIP global
vector is loaded independently of patch tokens.

Model checkpoints bind all three image-source IDs separately, as well as encoder
metadata. A scene change can reuse extraction caches but requires the matching
trained model or a new training run. Train-supervision keys depend on detections
and train GT, independently of scene/FAFA appearance features.

The FAFA baseline uses `grounding_dino_cache` and `cache/detections` in
`configs/fafa.yaml`. Build detections before its `prepare` command. After changing
its detection-cache contract, rerun baseline validation to create a fresh test
protocol lock. No labels, matching equations, selector thresholds or evaluator
definitions change with this separation.

Detection diagnostics require no scene or FAFA cache:

```bash
.venv-proposed/bin/python tools/data/audit.py \
  --detection-cache cache/detections \
  --image-root /path/to/images --output runs/detection_audit.json
```

`prepare-supervision --config configs/proposed.yaml` also needs only the shared
detection cache and train annotations. Detection auditing accepts only
`--detection-cache`; it never needs raw images or scene dimensions for alignment.
