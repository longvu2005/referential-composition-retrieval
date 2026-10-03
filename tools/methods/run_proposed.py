"""Optionally build/train, then retrieve and evaluate proposed on val/test.

Use the existing stage CLIs in separate processes so each releases its GPU memory.
Resolved YAML files are retained beside the results for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/methods/proposed"


def _path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load(path):
    with _path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _data_paths(cfg, args):
    for name in ("final_dir", "image_root", "cache"):
        cfg["data"][name] = str(_path(getattr(args, name) or cfg["data"][name]))


def _stage(script, cfg, config_path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "tools/methods" / script),
        "--config",
        str(config_path),
    ]
    print(f"Running {script}: {config_path}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _require_cache(cfg):
    root = _path(cfg["data"]["cache"])
    if (root / ".building").exists():
        raise RuntimeError(f"Cache build is incomplete: {root}; rebuild it first")
    if not (root / "index.pt").is_file():
        raise FileNotFoundError(
            f"Missing {root / 'index.pt'}; run scripts/build_cache.bash "
            "or use --build-cache --train"
        )


def _require_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run with --train or set --checkpoint")
    if not (path.parent / "tokenizer").is_dir():
        raise FileNotFoundError(
            f"Missing {path.parent / 'tokenizer'} for the checkpoint"
        )


def run_pipeline(args):
    if args.build_cache and not args.train:
        raise ValueError(
            "--build-cache requires --train: rebuilding changes the cache ID"
        )
    if args.train and args.checkpoint:
        raise ValueError(
            "--checkpoint is for evaluation only; --train evaluates its new best.pt"
        )
    splits = [s for s in ("val", "test") if s in args.splits]
    retrieve = _load(args.config)
    retrieve["method"] = "proposed"
    _data_paths(retrieve, args)
    training = _load(args.train_config) if args.train else None
    build = _load(args.cache_config) if args.build_cache else None
    for label, cfg in (("train", training), ("build_cache", build)):
        if cfg is None:
            continue
        _data_paths(cfg, args)
        if cfg["data"] != retrieve["data"]:
            raise ValueError(
                f"{label} and retrieval data/cache paths differ; "
                "align configs or use path overrides"
            )
    retrieval = retrieve["retrieval"]
    for key in ("top_m", "fine_batch_size", "identity_batch_size", "coarse_batch_size"):
        if int(retrieval.get(key, 512)) < 1:
            raise ValueError(f"retrieval.{key} must be positive")
    ks = retrieve.get("candidate_ks", [100, 500])
    if not ks or any(int(k) < 1 for k in ks) or max(ks) > int(retrieval["top_m"]):
        raise ValueError(
            "candidate_ks must be positive and cannot exceed retrieval.top_m"
        )
    retrieve["candidate_ks"] = ks
    if training is not None:
        if not training.get("evaluation", {}).get("enabled", False):
            raise ValueError(
                "--train requires evaluation.enabled to select best.pt on val"
            )
        if int(training["train"]["epochs"]) < 1:
            raise ValueError("train.epochs must be positive")
        if int(training["evaluation"]["top_m"]) != int(retrieval["top_m"]):
            raise ValueError("training evaluation.top_m must match retrieval.top_m")
        training["output"]["dir"] = str(
            _path(args.output_dir or training["output"]["dir"])
        )
        retrieve["checkpoint"] = str(Path(training["output"]["dir"]) / "best.pt")
        retrieve["output"]["dir"] = str(Path(training["output"]["dir"]) / "{split}")
    else:
        retrieve["checkpoint"] = str(_path(args.checkpoint or retrieve["checkpoint"]))
        if args.output_dir:
            retrieve["output"]["dir"] = str(_path(args.output_dir) / "{split}")
    # Require distinct paths even when only one split is requested today.
    template = retrieve["output"]["dir"]
    directories = {
        s: _path(template.format(split=s, method="proposed")) for s in ("val", "test")
    }
    if directories["val"] == directories["test"]:
        raise ValueError(
            "output.dir must distinguish val/test; use {split} or --output-dir"
        )
    summary_dir = (
        _path(args.output_dir) if args.output_dir else directories["val"].parent
    )
    config_dir = summary_dir / "runner_configs"
    checkpoint = Path(retrieve["checkpoint"])
    # Check these before writing outputs or launching expensive stages.
    if not args.build_cache:
        _require_cache(retrieve)
    if not args.train:
        _require_checkpoint(checkpoint)
    summary_path = summary_dir / "summary.csv"
    summary_path.unlink(missing_ok=True)
    if build is not None:
        _stage("build_cache.py", build, config_dir / "build_cache.yaml")
        _require_cache(retrieve)
    if training is not None:
        _stage("train_proposed.py", training, config_dir / "train.yaml")
        _require_checkpoint(checkpoint)
    rows = []
    for split in splits:
        cfg = copy.deepcopy(retrieve)
        cfg["split"] = split
        cfg["output"]["dir"] = str(directories[split])
        path = config_dir / f"{split}.yaml"
        _stage("retrieve_proposed.py", cfg, path)
        # Same resolved config and official evaluator as the baselines.
        _stage("evaluate.py", cfg, path)
        result = json.loads(
            (directories[split] / "metrics.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "method": "proposed",
                "split": split,
                "checkpoint": str(checkpoint),
                **result["overall"],
            }
        )
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(summary_path)
    print(f"Saved summary: {summary_path}", flush=True)
    return rows


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{CONFIG}/retrieve.yaml")
    parser.add_argument("--train-config", default=f"{CONFIG}/train.yaml")
    parser.add_argument("--cache-config", default=f"{CONFIG}/build_cache.yaml")
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Build cache before training (requires --train)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train from scratch, then evaluate the new best.pt",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=["val", "test"]
    )
    parser.add_argument(
        "--checkpoint",
        help="Evaluate this checkpoint; keep its sibling tokenizer/ directory",
    )
    parser.add_argument(
        "--output-dir", help="Result root; with --train also sets the training output"
    )
    parser.add_argument(
        "--final-dir", help="Override the final dataset path in all active stages"
    )
    parser.add_argument(
        "--image-root", help="Override the image root in all active stages"
    )
    parser.add_argument(
        "--cache", help="Override the visual cache path in all active stages"
    )
    return parser


def main():
    run_pipeline(make_parser().parse_args())


if __name__ == "__main__":
    main()
