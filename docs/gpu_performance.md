The training and retrieval paths use bounded chunks while preserving the model's
matching and scoring rules. Native FAFA targets are loaded only for training pairs
used by context losses, padded to each chunk's useful person columns, and then
transferred to the accelerator. Assignment is detached and computed once per
chunk; activation checkpointing recomputes the neural context path during backward.

Box suppression and SciPy assignment share a batched CPU workspace. This avoids
the former Python loop over CUDA scalar indices and dynamically sized masks.
Suppression retains stable detection-order ties, the strict IoU threshold and
greedy behavior; assignment retains shared one-to-one matching and private nulls.
Mining skips contextual attention/reasoning when its context coefficient is zero.
Validation still collects context evidence for coefficient calibration.

The supplied configuration includes these performance settings:

| Setting | Default in `configs/proposed.yaml` | Effect |
| --- | ---: | --- |
| `train.context_batch_size` | 64 | Bounds neural context work and native target transfer per chunk. |
| `retrieval.query_batch_size` | 8 | Batches text/query encoding and coarse scoring by Subject count. Output query order is preserved. |
| `retrieval.gpu_cache_mib` | 512 | Budget for resident projected gallery identities, masks and global states. |
| `cache.text_lru_mib` | 768 | CPU budget for frozen CLIP token states; the learned projection still runs each step. |

The gallery remains in chunks. GPU residency is used only when the projected
gallery fits both the configured budget and half the currently free device memory.
Otherwise retrieval retains CPU-backed chunks. Set `retrieval.gpu_cache_mib=0`
to disable residency. Existing configurations without `query_batch_size` keep
single-query retrieval; they can opt into batching explicitly.

Frozen text cache keys are built from CPU token IDs before device transfer.
Training metrics and fine-score validation flags are copied to CPU in groups.
These changes retain the configured numerical precision. CUDA training still uses
the existing AMP/GradScaler policy; FDA and geometry attention keep their FP32
regions, and retrieval keeps FP32 inference.

Skipping unused context pairs changes how many rows pass through dropout. Existing
weights remain compatible because parameter names/shapes and the scoring contract
are unchanged. Training trajectories can differ from older source. Resume should
use a compatible saved configuration and source as described in the run guide.

Measure the production joint training kernels on a CUDA machine without downloading
models or preparing a dataset:

```bash
.venv-proposed/bin/python tools/profile_gpu.py --device cuda --trace \
  --output-dir runs/gpu-profile
```

`profile.json` records synchronized step timings, throughput, peak allocated and
reserved VRAM, AMP scaler state and the number of context pairs/native person
slots processed. `trace.json` can be opened in a Chrome trace/Perfetto viewer.
Warmup iterations are excluded from timings; the optional traced step runs after
the timing loop so profiler overhead does not enter reported throughput.

The tool uses synthetic cached CLIP/FAFA feature shapes and the actual model,
loss, matching, activation checkpointing and optimizer. It excludes disk/image
I/O, frozen CLIP encoding, mining and validation, so its throughput is not an
end-to-end training estimate. Compare results with the same configuration, hardware,
software, batch/person sizes and AMP mode. `--people` controls person padding and
can be increased to exercise crowded images; `--no-amp` provides an FP32 control.

For a small CPU functional check:

```bash
PYTHONPATH=src:. python3.11 tools/profile_gpu.py --device cpu \
  --batch-size 2 --candidates 4 --people 2 --warmup 1 --steps 2 --threads 1
```

Correctness coverage includes dense-versus-sparse context loss and gradients,
dropout/checkpoint RNG, stable greedy suppression, duplicate/null assignments,
paired coarse-score gradients, deferred native reads and batched retrieval order.
CUDA-specific cases run automatically when CUDA is available and are skipped on
CPU-only development machines.
