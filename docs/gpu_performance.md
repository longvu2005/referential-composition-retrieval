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

For low utilization on an RTX 5060, check the runtime and the input pipeline
separately. The synthetic kernel profiler above excludes the cache reads and
assembly that can leave the GPU waiting. Run the following in the same Python
environment and with the same configuration as training:

```bash
.venv-proposed/bin/python tools/diagnose_gpu.py --device cuda \
  --config configs/proposed.yaml --cache-batches 3 \
  --output runs/gpu-diagnostics/joint.json
.venv-proposed/bin/python tools/diagnose_gpu.py --device cuda \
  --config configs/proposed.yaml --cache-batches 3 --epoch 1 \
  --output runs/gpu-diagnostics/warmup.json
```

Without `--cache-batches`, the tool only checks the environment. It records the
interpreter, PyTorch CUDA build, compiled architectures, CPU thread counts,
relevant debugging/thread environment variables, source status and an
`nvidia-smi` snapshot. A small CUDA forward/backward smoke test catches cases
where `is_available()` is true but the installed wheel cannot execute kernels.
Failure is saved in the JSON and returns a nonzero exit status without reading
the dataset. It does not install packages, download weights, train, or change
caches/checkpoints.

RTX 5060 has compute capability 12.0 (`sm_120`), according to
[NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus).
[PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/).
Use a wheel that supports this architecture. The proposed runtime allows
PyTorch >=2.6, which alone does not ensure Blackwell support; the separate
person-encoder environment pins PyTorch 2.5.1 and is not a supported Blackwell
runtime. Do not assume a recent driver fixes an older PyTorch wheel. The CUDA
version in `nvidia-smi` describes driver support; `torch.version.cuda` identifies
the PyTorch CUDA build. Changing the pinned FAFA environment needs separate
compatibility validation. Existing compatible feature caches can be used for
training without loading the native FAFA encoder.

The optional cache probe uses the actual readers, seeded training candidate
sampler, collation and native target chunk sizes. It times CPU assembly and
serial synchronized H2D transfers separately, counts reader LRU hits/misses,
and reports payload/padding. The default epoch is the first joint epoch;
`--epoch 1` measures warmup. Each batch is repeated twice by default. A new
reader starts with an empty Python LRU, but the OS file cache is not flushed.
Repeating an identical batch demonstrates the effect of LRU residency; it does
not estimate hit rates for a full epoch with changing candidates. The probe
excludes text/model execution, label/extra-crop preparation, matching, optimizer,
mining, validation and checkpoint I/O. Candidates do not use learned hard pools.
These component timings must not be presented as end-to-end training throughput.
Thread counts are preserved unless `--threads` is supplied.

The November 20, 2026 review of source based on `38e3276` found these remaining
constraints:

| Path | Evidence | Consequence |
| --- | --- | --- |
| Build-cache | Scene/detection extraction loops over individual images; FAFA uses batches of four crops within each image. | Decode/preprocessing and small forwards can leave gaps. This is separate from cached training and is not measured by either training-kernel or cache-reader timings. |
| Training input | Batch preparation runs in the train thread; visual buffers are pageable and copied on the current stream. | CPU assembly does not overlap the next batch with GPU compute. `non_blocking=True` alone does not provide that overlap. |
| Warmup | Target scenes/boxes were loaded even though every context loss was disabled. | Training now reads only query scenes/geometry and obtains target persons/global features from their indices. |
| Joint training | The first seeded epoch-5 batch has 768 candidates, 255 eligible context pairs, and 4,886 actual target people padded to 43,008 slots. | Visual target loading is still dense even though native/context work is sparse. The visual payload is 242.74 MiB, plus 90.23 MiB of native queries/targets in this example. |
| Matching | Detached assignment still transfers evidence to CPU for NumPy/SciPy and then returns indices to the device. | Each context/fine chunk has a synchronization boundary, despite the earlier removal of per-box synchronizations. |
| Validation | With 1,799 queries and `top_m=500`, the fine path scores 899,500 pairs in 14,392 chunks of at most 64. | Query batching improves coarse scoring; fine scoring still loops through each query's shortlist. |
| Mining | Context coefficient zero skips the neural reasoner, but the target helper still reads/transfers scene features. | Additional scene I/O remains even when mining only needs identity/FDA/coverage. |
| Calibration | The configured 8 x 12 x 6 grid evaluates 576 coefficient combinations on CPU. | GPU inactivity during coefficient selection is expected. |

The measured first joint batch on the development Mac/T9 took 6.54 seconds for
visual read/collation and 1.42 seconds for native reads/padding; the same batch
repeated with reader LRU hits took 0.171 and 0.031 seconds respectively. Its first
pass had 761 visual, 48 native-query and 239 native-target LRU misses. These are
observations on that host with PyTorch 2.2.2 and one CPU thread, without CUDA;
they do not establish RTX 5060 timings or steady-state disk performance.

For the same seeded epoch-1 batch, the warmup change reduces unique visual
reader requests from 767 to 24 and the visual transfer payload from 253.98 to
143.02 MiB. Query scene features, target person/global features and training
labels are retained. This is a work/payload reduction; no CUDA speedup has been
measured on the development host.

On the real machine, compare cache timings with the kernel profiler and collect
`nvidia-smi` while the slow stage is running:

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,pstate,temperature.gpu,power.draw,power.limit --format=csv -l 1
```

The diagnostic's single `nvidia-smi` snapshot is not a utilization trace.
[NVIDIA defines GPU utilization as time with at least one kernel executing](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization),
so a low value can reflect gaps waiting for CPU work. It is not the percentage
of CUDA cores used. For copy/compute overlap, PyTorch documents the need for
[pinned source memory and a separate CUDA stream](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
This pipeline has no DataLoader; adding a `num_workers` config key does not
introduce workers or prefetching. Next optimizations should measure/defer unused
joint/mining scenes, then prefetch with bounded buffers and correct stream
lifetimes. Preserve matching rules, numerical checks and the intended precision.

The warmup change preserves the loss/gradient contract and inference weights.
Resume verification also binds a hash of the Python source, so an existing run
must retain its original source; use the updated implementation for a new run.
