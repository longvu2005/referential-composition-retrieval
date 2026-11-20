"""Check the CUDA runtime and time real cache assembly without training/downloads."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from rcr.common.config import load_config
from rcr.common.data import (
    inference_samples,
    load_rcr_data,
    split_image_ids,
    split_samples,
)
from rcr.common.runtime import derived_generator
from rcr.dataset.audit import negative_exclusions
from rcr.proposed.batch import load_visual_features
from rcr.proposed.cache.store import GalleryCache, gallery_cache_options
from rcr.proposed.prefetch import PackedPersons
from rcr.proposed.sampling import (
    CandidateIndex,
    annotation_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)


def _command(arguments):
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def environment_report(cfg, device):
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_cuda_architectures": torch.cuda.get_arch_list(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "selected_device": str(device),
        "configured_device": cfg["runtime"]["device"],
        "amp": cfg["train"].get("amp", False),
        "prefetch_batches": cfg["train"].get("prefetch_batches", 0),
        "deterministic": cfg["runtime"].get("deterministic", True),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "source_commit": _command(["git", "rev-parse", "HEAD"]),
        "source_changes": _command(["git", "status", "--short"]),
    }
    smi = shutil.which("nvidia-smi")
    result["nvidia_smi"] = (
        _command(
            [
                smi,
                "--query-gpu=index,name,driver_version,memory.total,memory.used,"
                "utilization.gpu,pstate,temperature.gpu,power.draw,power.limit",
                "--format=csv",
            ]
        )
        if smi
        else {"error": "nvidia-smi is not on PATH"}
    )
    if device.type == "cuda":
        try:
            result["gpu"] = torch.cuda.get_device_name(device)
            result["compute_capability"] = list(
                torch.cuda.get_device_capability(device)
            )
            # is_available() alone does not prove this wheel can execute kernels.
            value = torch.randn(256, 256, device=device, requires_grad=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=result["amp"]):
                loss = (value @ value).float().square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            if not (torch.isfinite(loss) & value.grad.isfinite().all()).item():
                raise RuntimeError("CUDA forward/backward produced non-finite values")
            result["cuda_kernel_smoke"] = {"status": "passed"}
        except (RuntimeError, AssertionError) as error:
            result["cuda_kernel_smoke"] = {"status": "failed", "error": str(error)}
    else:
        result["cuda_kernel_smoke"] = {"status": "not run: selected device is CPU"}
    return result


class DiagnosticCache(GalleryCache):
    def __init__(self, **options):
        self.requests = Counter()
        super().__init__(**options)

    def _cached(self, key, load):
        native = isinstance(key, tuple)
        group = key[0] if native else "visual"
        items = self._native_items if native else self._items
        self.requests[f"{group}.{'hit' if key in items else 'miss'}"] += 1
        return super()._cached(key, load)


def _payload(tensors):
    return sum(value.numel() * value.element_size() for value in tensors) / 1024**2


def _transfer(tensors, device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    started = perf_counter()
    copies = [value.to(device, non_blocking=True) for value in tensors]
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    del copies
    return elapsed


def cache_report(cfg, device, *, batches, repeats, epoch):
    """Time production readers on seeded candidates; all cache files are read-only."""
    started = perf_counter()
    cache = DiagnosticCache(**gallery_cache_options(cfg))
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, "train")
    gallery = split_image_ids(data, "train")
    settings = sampling_settings(cfg["train"].get("sampling", {}))
    pools, partial = annotation_candidate_pools(
        data, samples, gallery, include_partial=settings["partial_id_quota"] > 0
    )
    index = CandidateIndex(
        samples, gallery, pools, negative_exclusions(samples), partial_pools=partial
    )
    generator = derived_generator(cfg["train"]["seed"], "sampling", epoch)
    plan = training_batches(
        samples,
        cfg["train"]["batch_size"],
        generator,
        case_balanced=cfg["train"].get("case_balanced", True),
    )
    setup_seconds = perf_counter() - started
    rows = []
    chunk_size = cfg["train"].get("context_batch_size", 64)
    context = cfg["loss"]["context_weight"] > 0 and epoch > cfg["train"].get(
        "context_warmup_epochs", 0
    )
    rejection = (
        cfg["loss"].get("context_rejection_weight", 0) > 0
        and epoch - 1 >= settings["warmup_epochs"]
    )
    for batch_index, samples in enumerate(plan[:batches]):
        sampled = sample_candidates(
            samples,
            cfg["train"]["candidates"],
            generator,
            index=index,
            positives_per_query=settings["positives_per_query"],
            identity_fraction=settings["identity_fraction"],
            hard_fraction=settings["hard_fraction"],
            semantic_hard_fraction=settings["semantic_hard_fraction"],
            partial_id_quota=settings["partial_id_quota"],
            identity_rejection=bool(
                cfg["loss"]["context_weight"]
                and cfg["loss"].get("context_rejection_weight", 0)
                and int((cfg["train"]["candidates"] - 1) * settings["hard_fraction"])
            ),
        )
        queries = torch.tensor([cache.by_id[s["query_image_id"]] for s in samples])
        targets = torch.tensor(
            [cache.by_id[image] for images in sampled.images for image in images]
        )
        eligible = (
            torch.tensor(
                [
                    image in pools[sample["sample_id"]]
                    or (rejection and image in sampled.hard_identity_ids[n])
                    for n, (sample, images) in enumerate(
                        zip(samples, sampled.images, strict=True)
                    )
                    for image in images
                ]
            )
            .nonzero()
            .flatten()
        )
        for repeat in range(repeats):
            previous = cache.requests.copy()
            started = perf_counter()
            query, target = load_visual_features(
                cache,
                queries,
                targets,
                target_context=context,
                defer_target_scenes=context,
                preserve_storage=True,
                pack_target_persons=True,
            )
            if context:
                target[0].prepare(eligible)
            packed = target[1]
            if not isinstance(packed, PackedPersons):
                packed = PackedPersons.from_padded(packed, target[3])
            assembly_seconds = perf_counter() - started
            tensors = [
                value
                for value in (
                    *query,
                    packed.values,
                    packed.positions,
                    target[2],
                    target[3],
                    cache.load_global_features(targets),
                    target[0].values if context else None,
                )
                if isinstance(value, torch.Tensor)
            ]
            counts = target[3].sum(1)
            result = {
                "batch": batch_index,
                "repeat": repeat,
                "queries": len(queries),
                "candidates": len(targets),
                "context_pairs": len(eligible) if context else 0,
                "partial_id_images": sampled.partial_id_count,
                "actual_target_people": int(counts.sum()),
                "padded_target_people": target[3].numel(),
                "projected_target_people": len(packed.values),
                "target_scene_pairs_loaded": len(eligible) if context else 0,
                "visual_cpu_assembly_seconds": assembly_seconds,
                "visual_payload_mib": _payload(tensors),
                "visual_h2d_seconds": _transfer(tensors, device),
                "visual_tensors_pinned": all(t.is_pinned() for t in tensors),
                "native_cpu_seconds": 0.0,
                "native_query_payload_mib": 0.0,
                "native_target_prefetch_mib": 0.0,
                "native_h2d_seconds": 0.0 if device.type == "cuda" else None,
            }
            if context:
                started = perf_counter()
                native = cache.native_queries(
                    inference_samples(samples), query[1].shape[1]
                )
                result["native_cpu_seconds"] += perf_counter() - started
                result["native_query_payload_mib"] += _payload([native])
                transfer = _transfer([native], device)
                if transfer is not None:
                    result["native_h2d_seconds"] += transfer
                del native
                started = perf_counter()
                prepared = cache.prepare_native_targets(
                    targets[eligible],
                    min(cfg.get("cache", {}).get("native_lru_mib", 0), 128) * 1024**2,
                )
                result["native_cpu_seconds"] += perf_counter() - started
                result["native_target_prefetch_mib"] = _payload(prepared.values())
                element_size = (
                    next(iter(prepared.values())).element_size()
                    if prepared
                    else cache.persons.element_size()
                )
                person_bytes = cache.native.tokens * cache.native.dim * element_size
                query_counts = query[3].sum(1)[eligible // cfg["train"]["candidates"]]
                matched_upper = int(torch.minimum(query_counts, counts[eligible]).sum())
                result["native_matched_transfer_upper_bound_mib"] = (
                    matched_upper * (person_bytes + 8) / 1024**2
                )
                old_slots = sum(
                    len(part) * int(counts[part].max())
                    for part in eligible.split(chunk_size)
                    if len(part)
                )
                result["previous_padded_native_transfer_mib"] = (
                    old_slots * person_bytes / 1024**2
                )
                del prepared
            result["cache_requests"] = dict(cache.requests - previous)
            rows.append(result)
            print(json.dumps(result), flush=True)
            del query, target, tensors
    return {
        "setup_seconds": setup_seconds,
        "epoch": epoch,
        "context_enabled": context,
        "scope": "production sparse cache preparation; raw serial visual/query H2D",
        "excludes": (
            "text/visual model forwards, labels/extra identity crops, loss, matching, "
            "matched target-token H2D, pinning/stream overlap, optimizer, mining, "
            "validation and checkpoint I/O"
        ),
        "sampling": "seeded candidates without learned hard pools; not a resumed batch",
        "repeat_policy": "same batch and reader LRU; OS disk cache is not flushed",
        "batches": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache-batches", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", default="runs/gpu-diagnostics/report.json")
    args = parser.parse_args(argv)
    if (
        args.cache_batches < 0
        or args.repeats < 1
        or (args.epoch is not None and args.epoch < 1)
        or (args.threads is not None and args.threads < 1)
    ):
        parser.error(
            "cache-batches must be nonnegative; repeats/epoch/threads positive"
        )
    cfg = load_config(args.config)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    result = {"environment": environment_report(cfg, device)}
    failed = result["environment"]["cuda_kernel_smoke"]["status"] == "failed"
    if args.cache_batches and not failed:
        try:
            result["cache"] = cache_report(
                cfg,
                device,
                batches=args.cache_batches,
                repeats=args.repeats,
                epoch=args.epoch or cfg["train"].get("context_warmup_epochs", 0) + 1,
            )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            result["cache"] = {"error": f"{type(error).__name__}: {error}"}
            failed = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["environment"], indent=2), flush=True)
    print(f"Diagnostics written to {output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
