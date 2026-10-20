"""Compare buffered/mmap cache reads without loading BERT/FAFA or training.

Run from the repository root in the proposed environment. Works on main before
this patch too. The same sampled batches are used for each mode. Files are read
only; OS page cache is uncontrolled, so both mode orders are measured.
"""

import argparse
import gc
import inspect
import json
import resource
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch

from rcr.common.config import load_config, override_config, parse_overrides
from rcr.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.dataset.audit import negative_exclusions
from rcr.proposed.batch import prepare_visual_batch
from rcr.proposed.cache.store import GalleryCache
from rcr.proposed.sampling import (
    CandidateIndex,
    identity_candidate_pools,
    sample_candidates,
    sampling_settings,
    training_batches,
)


@contextmanager
def read_mode(mmap_shards, scene_dir):
    """Keep large indices mmap; override only the fully consumed .pt shards."""
    original = torch.load
    counts = Counter()

    def load(path, *args, **kwargs):
        path = Path(path)
        if path.name != "index.pt" and path.parent.name in {"features", "targets"}:
            counts[f"{path.parent.parent.name}/{path.parent.name}"] += 1
        if path.parent == scene_dir or path.parent.name in {"targets", "queries"}:
            kwargs["mmap"] = mmap_shards
        return original(path, *args, **kwargs)

    torch.load = load
    try:
        yield counts
    finally:
        torch.load = original


def plan_batches(cfg, data, limit):
    train = cfg["train"]
    samples = split_samples(data, "train")
    gallery = split_image_ids(data, "train")
    pools = identity_candidate_pools(data, samples, gallery)
    excluded = negative_exclusions(samples)
    index = CandidateIndex(samples, gallery, pools, excluded)
    sampling = sampling_settings(train.get("sampling", {}))
    generator = torch.Generator().manual_seed(train["seed"])
    rows = training_batches(
        samples, train["batch_size"], generator,
        case_balanced=train.get("case_balanced", True),
    )[:limit]
    result = []
    for batch in rows:
        candidates = sample_candidates(
            batch, gallery, train["candidates"], generator,
            identity_pools=pools, excluded=excluded, index=index,
            identity_fraction=sampling["identity_fraction"],
            hard_fraction=sampling["hard_fraction"],
            positives_per_query=sampling["positives_per_query"],
        )
        result.append((batch, candidates))
    state = {key: set(value) for key, value in pools.items()}
    return result, state


def timed_method(obj, name, totals):
    original = getattr(obj, name)

    def measured(*args, **kwargs):
        start = perf_counter()
        result = original(*args, **kwargs)
        totals[name] = totals.get(name, 0) + perf_counter() - start
        return result

    setattr(obj, name, measured)


def profile(cfg, data, plan, state, mmap_shards, pass_number):
    cache_cfg = cfg.get("cache", {})
    paths = cfg["data"]
    with read_mode(mmap_shards, Path(paths["dino_cache"]) / "features") as reads:
        start = perf_counter()
        reader_options = {}
        if "native_lru_mib" in inspect.signature(GalleryCache).parameters:
            reader_options["native_lru_mib"] = cache_cfg.get("native_lru_mib", 0)
        cache = GalleryCache(
            paths["fafa_image_cache"], scene_root=paths["dino_cache"],
            query_root=paths.get("fafa_query_cache"),
            lru_mib=cache_cfg.get("lru_mib", 0), **reader_options,
        )
        index_seconds = perf_counter() - start
        supervision = None
        if "supervision" in inspect.signature(prepare_visual_batch).parameters:
            from rcr.proposed.supervision import TrainSupervision

            supervision_options = {}
            if "cache_file" in inspect.signature(TrainSupervision).parameters:
                supervision_options["cache_file"] = cache_cfg.get("supervision_cache")
            supervision = TrainSupervision(
                data, cache, scene_size=cfg["image_encoder"]["scene_size"],
                **supervision_options,
            )
        totals = {}
        for method in ("load_groups", "native_queries", "native_targets"):
            timed_method(cache, method, totals)
        if supervision is not None:
            timed_method(supervision, "load_ids", totals)
        source = cache._scene_cache
        print(json.dumps({
            "event": "mode", "pass": pass_number, "mmap_shards": mmap_shards,
            "index_seconds": index_seconds, "format_version": cache.format_version,
            "gallery_images": len(cache.image_ids), "patch_hw": list(cache.patch_hw),
            "lru_mib": cache_cfg.get("lru_mib", 0),
            "native_lru_mib": reader_options.get("native_lru_mib"),
            "supervision_cached_images": len(supervision._ids) if supervision else 0,
            "global_means_in_index": cache._global_features is not None or (
                source is not None and source._global_features is not None
            ),
            "scoped_global_reader": hasattr(cache, "load_global_features"),
        }), flush=True)
        before = resource.getrusage(resource.RUSAGE_SELF)
        for step, (rows, candidates) in enumerate(plan, 1):
            totals.clear()
            reads.clear()
            options = {"supervision": supervision} if supervision is not None else {}
            start = perf_counter()
            visual = prepare_visual_batch(
                rows, candidates, cache,
                state_image_ids=[state[row["sample_id"]] for row in rows], **options,
            )
            seconds = perf_counter() - start
            print(json.dumps({
                "event": "batch", "pass": pass_number, "mmap_shards": mmap_shards,
                "step": step, "seconds": seconds, "phases": dict(totals),
                "shard_loads": dict(reads),
                "tensor_mib": sum(t.numel() * t.element_size() for t in visual.values())
                / 1024**2,
                "target_dtype": str(visual["fafa_targets"].dtype),
            }), flush=True)
            del visual
        after = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps({
            "event": "memory", "pass": pass_number, "mmap_shards": mmap_shards,
            "peak_rss_mib": after.ru_maxrss / 1024,
            "minor_faults": after.ru_minflt - before.ru_minflt,
            "major_faults": after.ru_majflt - before.ru_majflt,
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--set", nargs="+", default=[])
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.batches < 1 or (args.threads is not None and args.threads < 1):
        parser.error("batch/thread counts must be positive")
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = override_config(load_config(args.config), parse_overrides(args.set))
    print(json.dumps({
        "event": "runtime", "python": sys.executable, "torch": str(torch.__version__),
        "threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda, "os_page_cache": "uncontrolled",
        "note": "CPU cache preparation only; no model forward/backward or GPU work",
    }), flush=True)
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    plan, state = plan_batches(cfg, data, args.batches)
    for pass_number, modes in enumerate(((False, True), (True, False)), 1):
        for mmap_shards in modes:
            profile(cfg, data, plan, state, mmap_shards, pass_number)
            gc.collect()


if __name__ == "__main__":
    main()
