# Independent person detection cache

Grounding DINO Tiny boxes are shared by Proposed, the standalone FAFA baseline,
train-only GT head alignment, and detection diagnostics. Scene extraction never
loads the detector; FAFA extraction never loads scene features.

| Config key | Default | Owns |
| --- | --- | --- |
| `data.detection_cache` | `cache/grounding-dino-tiny` | Ordered pixel boxes and original image sizes |
| `data.scene_cache` | `cache/proposed-clip-b32` | CLIP patches and native global vectors |
| `data.fafa_image_cache` | `cache/fafa-person-images` | Packed person means and projected target tokens |
| `data.fafa_query_cache` | `cache/fafa-person-queries` | Content-addressed multimodal query features |

All four roots must be distinct, including after resolving symlinks and parent
paths. Set `data.image_root` to the actual image directory before extraction.
Commands below run from the repository root after installing the environments.

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

## Import existing scene boxes without detector inference

For an existing complete unified CLIP or DINO scene cache:

```bash
.venv-proposed/bin/python - <<'PY'
from rcr.common.config import load_config
from rcr.proposed.cache.build import prepare_cache

cfg = load_config("configs/proposed.yaml")
prepare_cache(cfg, stage="detections", import_scene="/path/to/old-scene-cache")
PY
```

Import checks the source detector model/settings, gallery IDs/order, shard
identity and person counts. It copies pixel boxes without sorting or rounding.
If a legacy shard has only letterboxed boxes, its recorded scene size and raw
image header invert that transform. Images are needed to record original sizes.
No legacy identity labels are imported. Missing detector provenance is rejected.
The source directory is never modified. An unfinished legacy cache must be
completed before import. New and legacy cache directories cannot overlap.
An unfinished unified scene build cannot resume under the new scene-only
builder; finish it with the previous code or select a fresh scene directory.

A completed unified CLIP scene cache remains usable as `data.scene_cache`; its
geometry is ignored by the scene reader. Legacy DINO scene features still cannot
replace CLIP features. Old FAFA formats 4–6 bind a scene UUID and must be rebuilt
once in the new FAFA directories. They are not silently rebound to another crop
source. Historical diagnostics retain their explicit legacy reader.

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

The FAFA baseline uses `grounding_dino_cache` and `cache/grounding-dino-tiny` in
`configs/fafa.yaml`. Build detections before its `prepare` command. After changing
its detection-cache contract, rerun baseline validation to create a fresh test
protocol lock. No labels, matching equations, selector thresholds or evaluator
definitions change with this separation.

Detection diagnostics require no scene or FAFA cache:

```bash
.venv-proposed/bin/python tools/data/audit.py \
  --detection-cache cache/grounding-dino-tiny \
  --image-root /path/to/images --output runs/detection_audit.json
```

`prepare-supervision --config configs/proposed.yaml` also needs only the shared
detection cache and train annotations. Legacy `--cache` / `--scene-cache` audit
arguments are retained for historical unified caches.
