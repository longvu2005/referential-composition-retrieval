# Proposed method

The RCR model combines identity evidence with the requested visual change.
It uses a coarse retrieval stage followed by detailed scoring of shortlisted
images.

## Features

Grounding DINO provides person detections. Frozen CLIP encoders represent scene
and text inputs; frozen FAFA encodes person crops and text-conditioned person
features. Learned modules consume these features during training and retrieval.
See the [feature contracts](feature_contracts.md) for compatibility requirements.

## Grounding and composition

Grounding associates Subject descriptions with people in the reference image.
A Subject may contain one or several people. Composition combines these
representations with Subject roles and the requested change.

Membership and identity matching use predicted evidence at inference. Ground
truth annotations provide training supervision and evaluation labels.

## Retrieval and reasoning

Coarse retrieval combines identity similarity with global image-text state
similarity. The fine stage reranks the coarse shortlist using a shared partial,
one-to-one identity assignment across Subjects.

The fine score combines native FAFA evidence, identity coverage and contextual
reasoning. The reasoner evaluates individual Subjects and ordered Subject pairs.
Fine and coarse scores determine the final shortlist order; the remaining
gallery retains its coarse order.

## Objective and ranking

Training supervises grounding, person identity, coarse retrieval, contextual
reasoning and global state. Auxiliary losses, sampling and warmup are controlled
by [configs/proposed.yaml](../configs/proposed.yaml).

Validation selects the checkpoint and scoring coefficients. Test uses that
selection unchanged under the [benchmark protocol](benchmark_protocol.md).

## Configuration and checkpoints

Model and experiment settings live in YAML. Checkpoints record the effective
model configuration and feature provenance. Use the [run guide](proposed_runs.md)
for execution and the [cache guide](detection_cache.md) for feature preparation.
