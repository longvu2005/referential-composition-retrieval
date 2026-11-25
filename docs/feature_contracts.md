# Feature and checkpoint contracts

The proposed method uses frozen visual features and a pinned CLIP text tower
with optional LoRA adapters. Configuration and cache metadata identify the
encoders and preprocessing used by each run.

## Encoder features

| Source | Features |
| --- | --- |
| CLIP image encoder | Scene patches and global image embeddings |
| CLIP text encoder with optional LoRA | Token and global text embeddings |
| FAFA image encoder | Pooled person features and native target tokens |
| FAFA multimodal encoder | Text-conditioned query person features |

Scene geometry and person boxes must use compatible image transforms. CLIP and
FAFA keep their own text policies, recorded with the extracted features.
Encoder assets are defined in
[configs/person_encoder.yaml](../configs/person_encoder.yaml); experiment
settings are in [configs/proposed.yaml](../configs/proposed.yaml).

## Independent caches

Detection, scene, person-image and person-query caches have distinct directories
and identities. Readers check image order, encoder provenance, preprocessing and
feature compatibility before joining them.

Visual caches contain label-free features. Train annotation alignment is stored
separately and used for supervised learning. Completed compatible caches support
training and retrieval without loading the native FAFA model.

See the [cache guide](detection_cache.md) for construction and reuse.

## Checkpoint compatibility

Checkpoints bind the model architecture, effective settings, encoder metadata
and source cache identities. Retrieval restores the trained matching and pooling
settings. Validation-selected scoring coefficients are frozen for test together
with the checkpoint and validation fingerprints. Matching records the Subject
owner mode. Checkpoints without that field use `raw_logit` for retrieval.
The text checkpoint records the learned projection and any LoRA settings and
weights. Retrieval restores these from the checkpoint with the pinned CLIP base.
Checkpoints with frozen CLIP text use the projection alone.

Architecture or feature changes may require a fresh training run. Resume requires
the compatible full training state in `last.pt`; `best.pt` is for inference.
See the [run guide](proposed_runs.md) for both workflows.
