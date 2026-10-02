"""Run CLIP or FAFA on RCR and save complete rankings; no metric computation."""

import argparse
import importlib.metadata
import sys
from time import perf_counter

import torch
import yaml

from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.common.results import (
    output_directory,
    save_results,
    scores_to_rankings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("image", "text", "early_fusion", "late_fusion")
    )
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Smoke run on a prefix; gallery remains complete",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.mode:
        if cfg["method"] != "clip":
            parser.error("--mode is only for CLIP")
        cfg["mode"] = args.mode
    if args.split:
        cfg["split"] = args.split
    for key, value in cfg["runtime"].items():
        if key.endswith("batch_size") and int(value) < 1:
            parser.error(f"runtime.{key} must be positive")
    if int(cfg["runtime"].get("num_workers", 0)) < 0:
        parser.error("runtime.num_workers must be non-negative")
    cfg["output"]["dir"] = str(output_directory(cfg))
    data = load_rcr_data(cfg["data"]["final_dir"], cfg["data"]["image_root"])
    samples = split_samples(data, cfg["split"])
    full_num_queries = len(samples)
    if args.max_queries is not None:
        if args.max_queries < 1:
            parser.error("--max-queries must be positive")
        samples = samples[: args.max_queries]
    if not samples:
        parser.error("selected query split is empty")
    gallery_ids = split_image_ids(data, cfg["split"])
    device_name = cfg["runtime"]["device"]
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    print(
        f"{cfg['method']}: {len(samples)} queries / "
        f"{len(gallery_ids)} gallery images / {device}",
        flush=True,
    )
    start = perf_counter()
    if cfg["method"] == "clip":
        from rcr.methods.baselines.clip import retrieve_clip

        scores, details = retrieve_clip(data, samples, gallery_ids, cfg, device)
    elif cfg["method"] == "fafa":
        from rcr.methods.baselines.fafa import retrieve_fafa

        scores, details = retrieve_fafa(data, samples, gallery_ids, cfg, device)
    else:
        parser.error(f"unknown baseline {cfg['method']!r}")
    output = scores_to_rankings(samples, gallery_ids, scores)
    versions = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "transformers",
        "clip",
        "scipy",
        "timm",
        "eva-decord",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    save_results(
        cfg["output"]["dir"],
        output,
        {
            "method": cfg["method"],
            "mode": cfg.get("mode"),
            "config": cfg,
            "dataset_version": data.manifest.get("version"),
            "num_queries": len(samples),
            "num_gallery": len(gallery_ids),
            "query_subset": len(samples) != full_num_queries,
            "higher_is_better": True,
            "elapsed_seconds": perf_counter() - start,
            "python": sys.version,
            "packages": versions,
            **details,
        },
    )
    print(f"Saved {cfg['output']['dir']}/rankings.pt", flush=True)


if __name__ == "__main__":
    main()
