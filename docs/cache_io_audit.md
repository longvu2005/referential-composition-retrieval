# Cache I/O comparison

Compared `ec5a2a2b231545f92f0f8c3b05ffe9a6ad20b0a0` with
`692baf1b46e6c3e9501f9d1033e72c2b02e51f5e` on 20 October 2026.

## What is established

The attention/loss modules, dataset and training workload are unchanged between
these commits. A local CPU batch-read comparison does not reproduce a 48–96x
slowdown. It does expose a separate regression at the first mining/evaluation
that needs global image features: a packed FAFA index can trigger a scan of the
entire registered gallery through a legacy DINO source, despite ranking one split.

The original 16-hour run's runtime/progress data and real Kaggle cache are not
available in this environment. This audit does not establish that this global
pooling regression, mmap or CPU execution caused those 16 hours.

## Changed read paths

| Path | ec5a2a2 | Main before this patch |
| --- | --- | --- |
| FAFA persons | Dense index | Packed index also supported; same actual crops |
| Scene/native target shards | Buffered torch.load | mmap torch.load |
| LRU | Scenes only | Shared scene/target/query budget |
| Target batch | FP32 padded output | Storage dtype retained, usually FP16 |
| GT alignment | Labels in old feature shards | Fresh train-only alignment, memoized per image |
| Global image means | Stored in FAFA index | Packed FAFA relies on DINO; legacy DINO may have no means |

The shared LRU changes hit rates, not just retained bytes. It can evict more
scenes while avoiding target/query reads. An LRU hit on an mmap tensor does not
guarantee its pages remain resident: storage is brought in on access under OS
control. A warm local disk measurement cannot establish remote/mounted Kaggle
filesystem performance.

## Confirmed global-pooling regression

Before the patch, `_encode_gallery_state` first accesses `cache.global_features`.
If both packed FAFA and DINO lack stored means, this property reads every scene
before the requested gallery indices are selected. The snapshot contains 37,107
registered images; train mining only needs its 17,000 train images.

The patch calls `load_global_features(indices)` per gallery batch instead. Means
from existing indices are reused. Missing means are computed only for requested
images, reusing an existing scene LRU entry when possible, and retained as small
FP32 vectors for later mining/evaluation. Feature files stay immutable. The
explicit full-gallery property remains available for existing callers.

Default training first needs these gallery state vectors in mining at epoch 2;
epoch 1 has neither mining nor periodic validation. This regression cannot by
itself explain an epoch-1 training progress bar taking 16 hours.

In a synthetic 1,024-image cache, the projection requests 16 positions containing
13 unique images:

| Reader/cache | First scene reads | Repeat reads | First projection, local CPU |
| --- | ---: | ---: | ---: |
| ec5a2a2, dense index with stored means | 0 | 0 | 0.00084 s |
| Main, packed index + DINO without means | 1,024 | 0 | 0.47766 s |
| Patched, same packed cache | 13 | 0 | 0.00713 s |

The projected output hash is identical in all three cases. These times refer to
this one projection and synthetic local files, not an epoch or a T4 benchmark.

## Batch-read comparison and mmap

Six sampled batches each use 14 query images and 448 candidate positions over
the same synthetic 1,024-image working set. Scenes are 196x768 FP16; images have
1/2/4/8 persons and native targets are 32x256 FP16 per person. LRU is 256 MiB,
PyTorch is 2.5.1+cpu, one CPU thread. No models are loaded. OS page cache is not
controlled, and local file measurements vary with order and system load.

| Reader | Median batch preparation in the focused microbenchmark |
| --- | ---: |
| ec5a2a2, dense | 0.382 s |
| Main, dense | 0.304 s |
| Main, packed | 0.277 s |
| Main, packed + fresh train alignment | 0.363 s |

Across these six batches, old scene reads were 947 versus 1,082 for the shared
LRU; target reads fell from 2,163 to 1,124. Thus a shared LRU can increase scene
I/O while reducing total opens. The benchmark's smaller working set does not
predict the production hit rates.

Buffered versus mmap trials show no consistent large difference on this local
disk. The patch restores buffered scene/target reads as the default to match the
original I/O path; `cache.mmap_shards=true` retains the alternative for a real
filesystem comparison. Large indices remain mmap, packed storage and FP16 batches
remain supported. This is not proof that mmap caused the user's slowdown.

## Measure the real Kaggle cache

Run from the actual repository root with its proposed interpreter:

```bash
.venv-proposed/bin/python tools/profile_cache.py \
  --config configs/proposed.yaml --batches 1
```

The standalone copy in the patch package also works against main before applying
the patch. It prepares the exact same sampled CPU visual batch in each mode, in
both mode orders. It does not train, extract features, load BERT/FAFA weights or
change any cache file. Query text policy is read from the existing manifest; the
normal train-time policy check is not bypassed in the training code.

Each `batch` line reports total seconds; `load_groups`, `load_ids`,
`native_queries`, `native_targets`; shard-load counts; target dtype and batch
payload. Large `load_groups`/`native_targets` values identify the expensive
read/collate path. Large `load_ids` identifies first-visit train alignment/raw
image header access. Compare both passes because OS cache is uncontrolled.
`--threads 1` can be tested separately against the default CPU thread count.
The script's CUDA availability field describes that interpreter; it is not an
observation of another running training process.

## Validation and unchanged behavior

CPU suite: 626 passed, 10 skipped; skipped tests require CUDA. New tests reject
out-of-gallery scene reads, preserve requested order/duplicates, retain pooled
means after scene eviction, reuse scene LRU entries, avoid shard reads with
indexed means, and compare buffered/mapped scene/target values exactly.
Existing train/retrieve/evaluate and score-equivalence tests pass.

Cache IDs, crops, FAFA text policy, model architecture, losses, candidate counts,
mining schedule, validation splits and ranking order are unchanged. No cache
rebuild is required. No GPU-runtime or dependency-pinning changes are included.
