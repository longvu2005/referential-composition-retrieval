Training prepares a bounded batch ahead of the current step, transfers frozen
features in their stored precision, and runs the original loss and matching
rules. The implementation avoids transferring padded/unused evidence while
keeping model parameter names and shapes unchanged.

The supplied configuration enables these settings:

| Setting | Default | Effect |
| --- | ---: | --- |
| `train.prefetch_batches` | 1 | Prepares one future batch on a CPU worker. Zero disables background work. |
| `train.context_batch_size` | 64 | Bounds context forward/backward work per chunk. |
| `retrieval.query_batch_size` | 8 | Batches query/text encoding and coarse scoring by Subject count. |
| `retrieval.gpu_cache_mib` | 512 | Budget for resident gallery identity/mask/global projections. |
| `cache.text_lru_mib` | 768 | CPU budget for frozen CLIP token states; learned projections still run each step. |
| `cache.native_lru_mib` | 256 | Independent CPU LRU for native shards; batch prefetch retains at most 128 MiB of target tokens. |

Training uses the following input path:

1. The producer samples candidates using private CPU generators and prepares
   visual inputs/labels. Only query scenes are needed during warmup. Joint
   training reads target scene patches for eligible context pairs.
2. Person features are packed using the CPU validity mask. Only actual detections
   cross PCIe; their exact zero padding is reconstructed on the device. Target
   identity projection also skips padded detections. It contains no dropout.
3. With prefetch enabled on CUDA, the worker pins source buffers and copies frozen
   inputs on a separate CUDA stream. The consumer waits on an event and records
   the tensors on its stream. CPU source buffers are retained until consumption.
   Neural forwards, dropout, text projection and optimizer work stay on the
   training thread.
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
currently free CUDA memory; otherwise coarse scoring uses CPU-backed chunks.

No new downcast is introduced: FP16 stored features become FP32 model inputs on
the accelerator. FP32 stored features retain FP32 transfer. CUDA training retains
its AMP/GradScaler policy, and FDA/geometry attention retain their FP32 regions.
Packing can change floating-point accumulation through different GEMM shapes;
old and new training trajectories are not promised bitwise identical. Serial and
prefetched execution use the same numerical path and are tested against each other.

Cache files and inference weights remain compatible. Resume verification also
binds the source hash, so retain the original implementation for an existing run.
Use a new output directory with the updated implementation:

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
inputs. It reports raw visual payload, CPU assembly time, native prefetch bytes
and reader LRU requests. It measures serial visual/query H2D components without
pinning/stream overlap. Because it does not run grounding or matching, it reports
a one-to-one upper bound for matched native-token transfer, rather than claiming
to measure the actual assigned-token H2D path. It excludes label/extra-crop
preparation, text/model forwards, optimizer, mining, validation and checkpoint I/O.
Candidates use no learned hard pools. Each batch repeats twice by default; the
reader LRU warms but the OS file cache is not flushed. These component timings
are not end-to-end throughput or a full-epoch cache hit-rate estimate. CPU thread
counts are preserved unless `--threads` is supplied.

On the first seeded epoch-5 batch (24 queries, 32 candidates, 255 context pairs),
the source after the `a0b6246` review reduces work as follows:

| Quantity | Before this optimization | Updated path |
| --- | ---: | ---: |
| Visual transfer payload, excluding labels/extra crops | 242.74 MiB | 31.99 MiB |
| Target detections passed through identity projection | 43,008 padded slots | 4,886 actual detections |
| Target scene pairs loaded | 768 | 255 |
| Native target transfer | 89.67 MiB padded | At most 14.73 MiB, including positions |
| Native target read-ahead storage | Padded per chunk on the step thread | 21.78 MiB of raw token rows on the producer |

The native bound assumes every possible query detection is matched and is not
an actual assignment measurement. The payload values use real cache dimensions
and candidate ordering. The development machine has CPU-only PyTorch 2.2.2 on
macOS/T9; these results do not establish RTX 5060 speedups or utilization.

Measure the actual joint-training kernels on the CUDA machine:

```bash
.venv-proposed/bin/python tools/profile_gpu.py --device cuda --trace \
  --output-dir runs/gpu-profile
```

This runs the production model, joint loss, packed target identity projection,
matching, assigned-token transfer, activation checkpointing and optimizer on
synthetic frozen features. Warmup is excluded from timings. `profile.json`
records synchronized step times, throughput, transferred native person rows,
AMP scaler state and peak VRAM; the optional `trace.json` is collected after the
timing loop so tracing overhead does not enter reported throughput. It excludes
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

Regression coverage includes packed-vs-dense FDA and all pooling modes, null and
duplicate assignments, lazy scene reads outside backward, packed identity loss
and gradients under AMP/FP32, verified direction swaps, bounded prefetch/error
cleanup, serial-vs-prefetched training state and interrupted resume. CUDA stream
and numerical cases run when CUDA is available and skip on CPU-only hosts.
