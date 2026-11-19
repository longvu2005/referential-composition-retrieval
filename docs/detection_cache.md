# Feature caches

Feature extraction produces independent caches shared by training and retrieval.
Set their paths in [configs/proposed.yaml](../configs/proposed.yaml).

| Config key | Contents |
| --- | --- |
| `data.detection_cache` | Ordered person boxes and image sizes |
| `data.scene_cache` | Scene and global image features |
| `data.fafa_image_cache` | Person features and native target tokens |
| `data.fafa_query_cache` | Text-conditioned query features |
| `cache.supervision_cache` | Separate train annotation alignment |

The four feature directories must be distinct. Configure the raw image path and
person encoder environment before extraction.

## Build stages independently

Build all missing stages:

```bash
.venv-proposed/bin/python tools/run.py prepare --config configs/proposed.yaml
.venv-proposed/bin/python tools/run.py build-cache --config configs/proposed.yaml
```

Use `--cache-stage` to build an individual stage:

| Stage | Dependency |
| --- | --- |
| `detections` | Raw images and detector |
| `scene` | Raw images and scene encoder |
| `fafa-image` | Detections, raw images and FAFA assets |
| `fafa-query` | Detections, reference images, query text and FAFA assets |
| `fafa` | Both FAFA stages |

Detection and scene extraction can run independently. Re-run an interrupted
stage with the same settings to continue construction.

## Reuse and invalidation

Compatible completed caches can be reused across runs or mounted read-only.
Keep each cache's index and shards together when moving it, then update the
configured path.

Query text changes affect query features. Scene encoder changes affect scene
features. Changes to detections or person preprocessing affect dependent person
features. Readers validate stored identities and provenance before reuse.
Visual extraction and train annotation alignment remain separate.

## Supported detection inputs

The pipeline uses the independent pixel-box detection cache. Older combined
formats may require conversion with their original implementation or rebuilding.
Checkpoint and feature compatibility are described in the
[feature contracts](feature_contracts.md).
