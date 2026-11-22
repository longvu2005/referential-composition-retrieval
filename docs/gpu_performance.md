Training prepares a bounded batch ahead of the current step, transfers frozen
features in their stored precision, and runs the original loss and matching
rules. The input path avoids transferring padded/unused evidence. The aligned
CLS reasoner and its checkpoint version are described in the
[method guide](proposed_method.md).

The supplied configuration enables these settings:

| Setting | Default | Effect |
| --- | ---: | --- |
| `train.prefetch_batches` | 1 | Prepares one future batch on a CPU worker. Zero disables background work. |
| `train.context_batch_size` | 64 | Bounds context forward/backward work per chunk. |
| `retrieval.query_batch_size` | 8 | Batches query/text encoding and coarse scoring by Subject count. |
| `retrieval.prefetch_batches` | 1 | Reads one future fine-scoring chunk on a CPU worker; zero selects serial evaluation. |
| `retrieval.gpu_cache_mib` | 512 | Budget for resident gallery identity/mask/global projections. |
| `cache.text_lru_mib` | 768 | CPU budget for frozen CLIP token states; learned projections still run each step. |
| `cache.native_lru_mib` | 256 | Independent CPU LRU for native shards; batch prefetch retains at most 128 MiB of target tokens. |

Training uses the following input path:

1. The producer samples candidates using private CPU generators and prepares
   visual inputs/labels. Only query scenes are needed during warmup. Joint
   training reads target scene patches for eligible context pairs.
2. Person features are gathered directly from the packed CPU index. The producer
   does not allocate a dense target-person tensor only to pack it again. One
   batched gather retains crop order, repeated images and stored precision; boxes
   are collated together and remain small padded metadata. Only actual detections
   cross PCIe; their exact zero padding is reconstructed on the device. Target
   identity projection also skips padded detections. It contains no dropout.
   Query membership and identity labels are built in bulk, avoiding a tensor
   assignment for every annotated person/Subject. Public dense cache readers and
   third-party readers retain their existing API and fallback path.
3. With prefetch enabled on CUDA, the worker pins source buffers and copies frozen
   inputs on a separate CUDA stream. The consumer waits on an event and records
   the tensors on its stream. CPU source buffers are retained until consumption.
   Neural forwards, dropout, text projection and optimizer work stay on the
   training thread.
   Small masks/labels share pinned transfer packets by dtype; larger visual
   payloads retain their stored precision and bounded buffers. Consumer stream
   lifetimes are recorded once per shared allocation.
4. CPU layout metadata provides eligible pair positions and useful person widths,
   avoiding GPU-to-CPU reads of invariant masks. Assignment remains detached and
   runs once per chunk in its NumPy/SciPy workspace.
5. Native FAFA tokens are gathered using the CPU assignment already computed by
   matching. Only assigned token rows and their positions cross PCIe; null rows
   and person padding are reconstructed on the device before FP32 FDA scoring.
   Backward checkpointing recomputes only the neural context path.

Each prefetched batch carries the sampling/identity generator states following
that batch. Checkpoints commit the states of the consumed batch, even when the
worker has already sampled the next one. Cancellation and errors join the worker;
cache shard I/O is performed outside the LRU accounting lock. Native target
prefetch and scene staging are bounded by the batch and configured memory limits.
Prefetch retains inputs for one future batch, adding bounded CPU/pinned memory
and CUDA input buffers. Set `train.prefetch_batches=0` when measuring the serial
control or when that extra input batch exceeds the available VRAM.

Retrieval preserves query output order and FP32 inference. Mining with context
coefficient zero skips the reasoner and target scene loading in separated caches.
Fine scoring uses the same packed assigned-token path as training. Validation
still collects context evidence for coefficient calibration. Gallery residency is
used only when the projections fit both the configured budget and half the
available CUDA memory, including reusable blocks held by the PyTorch allocator;
otherwise coarse scoring uses CPU-backed chunks.

Fine evaluation loads person geometry first, packs actual detections in their
stored precision and projects each actual person once for semantics and
identity. Candidates are grouped by person count to reduce padding; component
scores are scattered back into the original Top-M order before fusion. The
detached assignment determines which candidates can contribute context. Only
candidates with a match load target scenes and native FAFA rows or enter the
reasoner. An all-missing candidate has exact `H=0`, `C=0` and the configured
negative missing penalty for every native pooling mode. This shortcut also
applies during early validation when many predicted assignments are empty.
For matched candidates, evaluation also removes masked reference keys and
masked reasoner rows/keys before attention. Every selected GROUP member,
missing-match member and active ordered pair remains present. Geometry is
gathered on the same person slots. This is the same masked attention population
with less matrix work; training keeps its full layout and dropout path.
Validation still computes raw `H` when its coefficient is zero. Base mining
skips the reasoner for semantic candidates. Rejection context mining computes
`H` only for the bounded wrong-ID shortlist and reuses its `B,C` in the base
union, avoiding a second assignment or binding pass. Setting
`train.sampling.rejection_mining_context_weight=0.0` skips the reasoner for all
mining candidates.

Fine-scoring prefetch is bounded to one geometry/packed-person chunk. It runs
no model forward or CUDA kernel on the worker. CUDA sources are pinned there;
cache errors propagate and the worker joins on every exit. `prefetch_batches=0`
uses the same scoring path without a background worker. Lightweight third-party
readers without the separate geometry/scene API retain the public dense scorer.
Normalized detection boxes share the existing visual LRU between matching and
scene loads, avoiding duplicate geometry-shard reads without another cache
format or an additional memory budget.

CPU calibration batches normalization by valid support size and computes AP
from compact positive ranks. The sequential float64 accumulation includes the
fixed coarse tail in official rank order, retaining exact AP values and the
first-candidate tie rule. The complete coefficient grid, Top-M population and
checkpoint criterion are unchanged. CPU replay/calibration still work without
model imports, weights or feature caches.

Retrieval logs wall times for gallery encoding, query/coarse scoring, fine
scoring and ranking, and saves them as `timings_seconds` in `rankings.pt` and
`run.json`. These are elapsed stage times including I/O and synchronization at
output boundaries, not individual CUDA kernel timings. Calibration prints its
candidate count and elapsed CPU time and adds a `calibration` timing to selected
artifacts. Compare the same checkpoint, split and runtime when measuring an
optimization; use a separate output directory:

```bash
.venv-proposed/bin/python tools/run.py run --config runs/proposed/selected.yaml \
  --split val --set retrieval.prefetch_batches=0 output.dir=runs/eval-serial
.venv-proposed/bin/python tools/run.py run --config runs/proposed/selected.yaml \
  --split val --set retrieval.prefetch_batches=1 output.dir=runs/eval-prefetch
```

Serial and prefetched evaluation use the same checkpoint and score-artifact
schema for both v5 and v4 inference.

CPU/GPU boundaries are intentional and bounded as follows:

| Stage | Readback policy |
| --- | --- |
| Query box suppression | One packed predicted confidence/geometry read for the CPU suppression workspace; zero/one-person tensors skip suppression/readback. |
| Identity crop mining | Known annotation positions stay on CPU; only the model-dependent hardness vector is read when hard mining is active. |
| Fine matching | One packed similarity/mask/geometry read per chunk for exact SciPy assignment; context/swap flags share that read. CPU assignments also drive native-token gathers. |
| Train checks and metrics | One FP32 packet per step, validated before backward or optimizer updates. Nested losses add device predicates without Python truth-value checks. |
| Gradient logging | Model/text norms share one read at the configured log interval. AMP optimizer overflow checks retain their framework behavior. |
| Coarse retrieval/mining | Identity and state scores share one read per query batch; both use the same CPU shortlist/fusion convention. |
| Fine retrieval/mining | H, C, all native poolings and pending numerical checks share one read per query. |
| Gallery projection | Residency is decided from CPU masks and the VRAM budget before projection. Resident outputs stay on GPU; the fallback reads identity/state together per chunk. |
| Frozen text cache | Misses write CPU cache rows once while current states stay on GPU; hits upload only unique cached rows in one batch. CPU token IDs stay on CPU for cache keys. |
| Checkpoints | Commit the consumed training stream, without fencing the independent producer copy stream. Tensor serialization still performs its required state copies. |

CPU annotation layouts replace CUDA boolean-index operations whose output size
would require a host wait. A batch layout checks tensor identity/mutation
versions and falls back when callers edit its supervision. The identity loss
masks unknown observations at fixed size, including the empty/sole-known cases,
without a CUDA nonzero operation. Swap annotations remain on CPU and reuse
selection flags and assignments already returned by matching. Both original
and edited sentences still use the exact same assignment and neural scorer.

Standalone score/loss APIs continue to reject invalid inputs before returning.
Training collects their numerical predicates and rejects failures before any
backward/optimizer side effect. CPU metadata checks, necessary assignment reads,
cache persistence and checkpoint serialization remain explicit; removing a
`.tolist()` from CPU-only bookkeeping would not remove a GPU wait.

No new downcast is introduced: FP16 stored features become FP32 model inputs on
the accelerator. FP32 stored features retain FP32 transfer. CUDA training retains
its AMP/GradScaler policy, and FDA/geometry attention retain their FP32 regions.
Packing can change floating-point accumulation through different GEMM shapes;
old and new training trajectories are not promised bitwise identical. Serial and
prefetched execution use the same numerical path and are tested against each other.

Training resume binds the architecture and source hash. Keep the original
implementation to resume an existing run. To train with the updated model
and compatible frozen caches, use a new output directory.

```bash
.venv-proposed/bin/python tools/run.py train --config configs/proposed.yaml \
  --set output.dir=runs/proposed-gpu-optimized
```

For an RTX 5060, first check the exact Python/CUDA runtime. NVIDIA lists the
[RTX 5060 at compute capability 12.0 (`sm_120`)](https://developer.nvidia.com/cuda/gpus).
[PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/).
The proposed requirement `torch>=2.6` alone does not ensure Blackwell support;
the separate FAFA environment pins PyTorch 2.5.1 and is not a supported Blackwell
runtime. A recent driver does not replace an older PyTorch wheel. Existing
compatible feature caches allow training without loading that frozen FAFA runtime.
Changing its pinned dependencies requires separate encoder compatibility checks.

Run diagnostics in the same environment/configuration as the slow training:

```bash
.venv-proposed/bin/python tools/diagnose_gpu.py --device cuda \
  --config configs/proposed.yaml --cache-batches 3 \
  --output runs/gpu-diagnostics/joint.json
.venv-proposed/bin/python tools/diagnose_gpu.py --device cuda \
  --config configs/proposed.yaml --cache-batches 3 --epoch 1 \
  --output runs/gpu-diagnostics/warmup.json
```

Without `--cache-batches`, diagnostics only inspect the runtime. The JSON records
PyTorch/CUDA versions, compiled architectures, interpreter, CPU thread counts,
relevant environment variables, source status and an `nvidia-smi` snapshot. A
small CUDA forward/backward smoke catches unsupported kernels even when
`torch.cuda.is_available()` returns true. Failures are saved and return a nonzero
exit status. No weights are downloaded and no caches/checkpoints are modified.

The cache probe reads actual seeded candidates, sparse scenes and packed person
inputs, including the configured partial-ID top-up. It uses the same direct
packed readers as training, coarse gallery encoding and fine scoring. It reports
raw visual payload, CPU assembly time, native prefetch bytes
and reader LRU requests. It measures serial visual/query H2D components without
pinning/stream overlap. Because it does not run grounding or matching, it reports
a one-to-one upper bound for matched native-token transfer, rather than claiming
to measure the actual assigned-token H2D path. It excludes label/extra-crop
preparation, text/model forwards, optimizer, mining, validation and checkpoint I/O.
Candidates use no learned hard pools. Each batch repeats twice by default; the
reader LRU warms but the OS file cache is not flushed. These component timings
are not end-to-end throughput or a full-epoch cache hit-rate estimate. CPU thread
counts are preserved unless `--threads` is supplied.

Measure the actual joint-training kernels on the CUDA machine:

```bash
.venv-proposed/bin/python tools/profile_gpu.py --device cuda --trace \
  --output-dir runs/gpu-profile
```

This runs the production model, joint loss, packed target identity projection,
matching, assigned-token transfer, activation checkpointing and optimizer on
synthetic frozen features. Warmup is excluded from timings. `profile.json`
records synchronized step times, throughput, transferred native person rows,
AMP scaler state and peak VRAM. With `--trace`, `transfer_audit` also counts
explicit transfer calls/source bytes, scalar reads and dynamic-shape operations
by device. The optional `trace.json` is collected after the timing loop so audit
and tracing overhead do not enter reported throughput. The audit runs in a
separate extra step so dispatch instrumentation cannot duplicate trace events.
CUDA runtime/internal
synchronizations are visible in the trace; source bytes are logical payloads,
not measured PCIe traffic. It excludes
real cache assembly, frozen CLIP forward, producer overlap, mining and validation.
Compare the same workload/configuration, hardware, software, person counts and
AMP mode. `--people` controls crowded-image padding; `--no-amp` gives an FP32
control. For a CPU functional check:

```bash
PYTHONPATH=src:. python3.11 tools/profile_gpu.py --device cpu \
  --batch-size 2 --candidates 4 --people 2 --warmup 1 --steps 2 --threads 1
```

Collect utilization while the real slow stage is running; the diagnostic's
single snapshot is not a utilization trace:

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,pstate,temperature.gpu,power.draw,power.limit --format=csv -l 1
```

[NVIDIA defines GPU utilization as time with at least one kernel executing](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization).
Low utilization can therefore reflect CPU gaps rather than GPU compute limits.
[PyTorch documents pinned sources and separate streams for copy/compute overlap](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
The updated CUDA producer uses both, with explicit event and buffer lifetimes.
There is no DataLoader; adding an unrelated `num_workers` setting has no effect.

Remaining stage-specific constraints include CPU SciPy assignment per chunk,
individual image/crop processing during build-cache, and CPU calibration of the
configured 576 coefficient combinations. GPU inactivity during calibration is
expected. Further changes to assignment, precision, compilation or batch sizes
should be guided by a trace on the actual RTX 5060 rather than utilization alone.

Regression coverage includes grouped checks/error propagation, fixed-shape
identity loss and CPU layout invalidation, matching metadata piggybacking,
direct gallery residency and coalesced score outputs, packed-vs-dense FDA and all pooling modes, null and
duplicate assignments, lazy scene reads outside backward, packed identity loss
and gradients under AMP/FP32, verified direction swaps, bounded prefetch/error
cleanup, serial-vs-prefetched training state and interrupted resume. CUDA stream
and numerical cases run when CUDA is available and skip on CPU-only hosts.
