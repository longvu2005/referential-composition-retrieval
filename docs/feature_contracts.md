# Proposed: frozen feature and checkpoint contracts

The proposed method uses frozen CLIP image/text encoders and a separate frozen
FAFA person encoder. These contracts describe the current implementation;
run commands live in [proposed_runs.md](proposed_runs.md).

## CLIP image and text features

Both encoders use `openai/clip-vit-base-patch32`, pinned to revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

| Feature | Shape per image/text | Use |
| --- | --- | --- |
| Scene patches | `49 x 768` (7 x 7 grid) | Learned context projection to width 384 |
| Projected image CLS | `512` | Global state; never replaced by a patch mean |
| Projected text tokens | Up to `77 x 512` | Learned context projection to width 384 |
| Projected text EOT | `512` | Global state |

Scene inputs are complete images letterboxed to 224 x 224 with bicubic resize,
then CLIP rescaling/normalization. There is no center crop. Detector pixel boxes
are mapped using the same letterbox geometry when caches are joined.
Frozen native projections stay frozen; the additional context projections learn.
The 512-D state image/text adapters start as identity maps with zero bias, so
their initial cosine is native CLIP cosine. State supervision then adapts them.

CLIP keeps its native vocabulary and 77 positions, including BOS/EOS. Character
offsets locate complete Subject mentions for role injection after the frozen
text backbone. No special vocabulary entries are added. Truncation preserves
EOS and records statistics; losing every complete mention of a required Subject
raises an error. Training saves `clip_text_stats` in its effective configuration;
retrieval saves query-scoped statistics in `rankings.pt` and `run.json`.

## FAFA person features and text policy

The image cache stores mean-pooled image-only Q-Former hidden states before
`vision_proj`, plus projected native target tokens. Query features come from
the separate text-conditioned multimodal path. The model's learned person
projection consumes the pooled hidden states; native FDA consumes the native
query and target representations. See [method equations](proposed_method.md).

Native FAFA keeps its separate `checkpoint.max_txt_len: 128` policy, including
special tokens, and a caption word cap at least as large. Manifests record the
policy; older manifests/checkpoints without it retain the pinned 32-token,
50-word meaning. A policy change re-extracts query features only. Interrupted
builds must resume with their original policy or use a separate directory.
Train/retrieve validate the native YAML when available; cache-only machines
can rely on the recorded policy without installing native FAFA.

## Independent caches

| Config key | Default path |
| --- | --- |
| `data.detection_cache` | `cache/detections` |
| `data.scene_cache` | `cache/scene` |
| `data.fafa_image_cache` | `cache/person-images` |
| `data.fafa_query_cache` | `cache/person-queries` |
| `cache.supervision_cache` | `cache/train-supervision.json` (train labels only) |

Names describe each cache's role; encoder models and revisions remain recorded
in the cache metadata. The FAFA baseline reads the same `cache/detections`.

FAFA features bind detection identity and crop order, independently of the scene
encoder. Cache validation checks source identities, checkpoint/revision,
preprocessing and separate feature widths. Completed unified CLIP scene caches
remain readable as scene inputs. Historical boxes require conversion with their
original code revision or a fresh detection build; the current runtime has no
scene-to-detection import command. Legacy scene-bound FAFA caches require
rebuilding in the format-7 independent directories. See the
[detection cache guide](detection_cache.md).

Cache indices carry their own identities; changing the root in YAML does not
modify the stored identity. Point the YAML at an existing compatible cache to
reuse it. Renaming defaults does not move or rebuild local artifacts.

## Checkpoint compatibility

New checkpoints use architecture name `fafa-partial-match-v2`. Retrieval also
accepts the previous CLIP-prefixed name for this exact architecture; only that
explicit alias is accepted. Tensor layouts, state keys and equations are
unchanged, including the recorded `binding_log_scale`. Existing compatible
checkpoints do not need conversion or retraining because of the name cleanup.

Historical DINO/BERT architecture checkpoints remain incompatible and are
rejected before loading text weights. Matching feature dimensions alone do
not make them compatible. Model checkpoints still bind scene, detection and
FAFA source identities and encoder metadata.

Current YAML outputs are `runs/proposed` and
`runs/proposed-calibration`. Existing run configs retain their explicit paths;
use those configs to read old runs.
Keep selected configurations and checkpoint bytes intact: test selection binds
the checkpoint SHA256 and validation fingerprint. For a new experiment, use a
fresh output directory and calibrate its validation artifact separately.

## Verification scope

CPU tests cover training, retrieval, calibration, full-gallery replay and native
CLIP behavior using tiny Transformers models without pretrained downloads.
They also compare scores from the current and previous checkpoint names.
A full GPU experiment requires raw images, pretrained assets and compatible
caches; CPU checks do not establish benchmark mAP.
