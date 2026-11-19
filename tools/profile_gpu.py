"""Profile training kernels on synthetic frozen features, without downloads."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from rcr.common.config import load_config
from rcr.common.runtime import seed_everything
from rcr.proposed.cache.native import NativeTargetBatch
from rcr.proposed.losses import compute_loss, loss_settings
from rcr.proposed.nn.model import RCRModel


class FrozenTargets:
    def __init__(self, values):
        self.values = values
        self.loaded_people = 0

    def native_targets(self, indices, width):
        self.loaded_people += len(indices) * width
        return self.values[:, :width].index_select(0, indices)


def _inputs(cfg, queries, candidates, people, device):
    """Mirror cached CLIP/FAFA shapes; annotations only determine loss masks."""
    scene_dim, person_dim, global_dim, native_dim = 768, 768, 512, 256
    tokens, patches, subjects, length = 32, 49, 2, 16
    query = torch.randn(queries, people, person_dim)
    target = torch.randn(queries, candidates, people, person_dim)
    same_count = min(candidates - 1, max(2, int(candidates * 0.5)))
    target[:, :same_count] = (
        query[:, None] + torch.randn_like(target[:, :same_count]) * 0.02
    )
    positive = torch.zeros(queries, candidates, dtype=torch.bool)
    positive[:, : min(2, same_count - 1)] = True
    same = torch.zeros_like(positive)
    same[:, :same_count] = True
    hard = torch.zeros_like(positive)
    hard[:, same_count : same_count + max(1, candidates // 10)] = True
    labels = torch.arange(queries * people).reshape(queries, people)
    target_labels = torch.full((queries, candidates, people), -1)
    target_labels[:, :2] = labels[:, None]
    target_mask = torch.arange(people) < torch.randint(
        1, people + 1, (queries, candidates, 1)
    )
    target_mask[:, :same_count] = True
    boxes = torch.rand(queries, people, 4)
    boxes[..., 2:] = boxes[..., :2] + 0.1
    target_boxes = boxes[:, None].expand(-1, candidates, -1, -1).clone()
    membership = torch.zeros(queries, subjects, people)
    membership[:, 0, ::2] = 1
    membership[:, 1, 1::2] = 1
    positions = torch.tensor([2, 8]).expand(queries, -1)
    mentions = torch.zeros(queries, subjects, length, dtype=torch.bool)
    mentions.scatter_(2, positions[..., None], True)
    batch = {
        "query_scene": torch.randn(queries, patches, scene_dim),
        "query_persons": query,
        "query_boxes": boxes,
        "query_person_mask": torch.ones(queries, people, dtype=torch.bool),
        "query_identity_labels": labels,
        "query_identity_mask": torch.ones_like(labels, dtype=torch.bool),
        "grounding_targets": membership,
        "target_scene": torch.randn(queries, candidates, patches, scene_dim),
        "target_persons": target,
        "target_boxes": target_boxes,
        "target_mask": target_mask,
        "target_global": torch.randn(queries, candidates, global_dim),
        "target_identity_labels": target_labels,
        "target_identity_mask": positive[..., None] & target_mask,
        "positive_mask": positive,
        "identity_positive_mask": same,
        "hard_identity_negative_mask": hard,
        "fafa_query": torch.randn(queries, subjects, people, native_dim),
        "subject_pos": positions,
        "subject_token_mask": mentions,
        "subject_ids": torch.tensor([1, 2]).expand(queries, -1),
        "change_global": torch.randn(queries, global_dim),
        "selection_mask": torch.ones(queries, subjects, length, dtype=torch.bool),
        "change_mask": torch.ones(queries, length, dtype=torch.bool),
    }
    batch = {key: value.to(device) for key, value in batch.items()}
    source = FrozenTargets(
        torch.randn(
            queries * candidates, people, tokens, native_dim, dtype=torch.float16
        )
    )
    batch["fafa_targets"] = NativeTargetBatch(
        source, torch.arange(queries * candidates)
    )
    batch["fafa_pooling"] = cfg["retrieval"]["pooling"]
    text = (
        torch.randn(queries, subjects, length, global_dim, device=device),
        torch.randn(queries, length, global_dim, device=device),
    )
    return batch, source, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--people", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output-dir", default="runs/gpu-profile")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    queries = cfg["train"]["batch_size"] if args.batch_size is None else args.batch_size
    candidates = (
        cfg["train"]["candidates"] if args.candidates is None else args.candidates
    )
    if (
        min(queries, args.people, args.steps, args.threads) < 1
        or candidates < 3
        or args.warmup < 0
    ):
        parser.error(
            "sizes/steps/threads must be positive, candidates >= 3, warmup >= 0"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; use --device cpu for a functional smoke check"
        )
    torch.set_num_threads(args.threads)
    seed_everything(
        cfg["train"]["seed"], deterministic=cfg["runtime"].get("deterministic", True)
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = (
        RCRModel.from_config(
            cfg["model"], input_dim=768, person_input_dim=768, global_dim=512
        )
        .to(device)
        .train()
    )
    projection = nn.Linear(512, cfg["model"]["dim"]).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": cfg["optimizer"]["lr"],
            },
            {"params": projection.parameters(), "lr": cfg["optimizer"]["text_lr"]},
        ],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    amp = device.type == "cuda" and not args.no_amp and cfg["train"].get("amp", False)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=amp)
    )
    batch, source, text = _inputs(cfg, queries, candidates, args.people, device)
    settings = loss_settings({**cfg["loss"], "context_swap_weight": 0.0})

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        # Match training: the learned text projection is outside visual AMP.
        batch["selections"], batch["change"] = (projection(value) for value in text)
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            loss, _ = compute_loss(
                model,
                batch,
                (7, 7),
                context_batch_size=cfg["train"]["context_batch_size"],
                context_score_weight=cfg["retrieval"]["context_weight"],
                coverage_score_weight=cfg["retrieval"]["coverage_weight"],
                **settings,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    synchronize()
    source.loaded_people = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(args.steps):
        started = perf_counter()
        loss = step()
        synchronize()
        timings.append(perf_counter() - started)
    if not torch.isfinite(loss):
        raise FloatingPointError("profiling produced a non-finite loss")
    result = {
        "workload": (
            "synthetic frozen features; production joint loss, matching, "
            "checkpoint and optimizer"
        ),
        "excludes": "image/disk I/O, frozen CLIP forward, mining and validation",
        "device": str(device),
        "torch": torch.__version__,
        "amp": amp,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "queries": queries,
        "candidates": candidates,
        "max_people": args.people,
        "context_batch_size": cfg["train"]["context_batch_size"],
        "context_pairs_per_step": int(
            (
                batch["identity_positive_mask"] | batch["hard_identity_negative_mask"]
            ).sum()
        ),
        "native_padded_people_per_step": source.loaded_people / args.steps,
        "warmup": args.warmup,
        "steps": args.steps,
        "step_seconds": timings,
        "median_step_seconds": statistics.median(timings),
        "queries_per_second": queries / statistics.median(timings),
        "final_loss": float(loss),
        "grad_scaler": scaler.state_dict(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2
        if device.type == "cuda"
        else None,
    }
    (output / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True
        ) as profile:
            step()
            synchronize()
        profile.export_chrome_trace(str(output / "trace.json"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
