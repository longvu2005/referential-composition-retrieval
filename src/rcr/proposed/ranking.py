"""Chunked identity shortlist and exact fine scores; no coarse/fine fusion."""

from time import perf_counter

import torch
from tqdm import tqdm

from rcr.common.data import split_image_ids
from rcr.proposed.batch import query_inputs, to_device
from rcr.proposed.scores import coarse_scores


def retrieval_settings(cfg):
    settings = {
        "mode": "shortlist",
        "top_m": 500,
        "fine_batch_size": 16,
        "identity_batch_size": 512,
        "coarse_batch_size": 256,
    }
    if set(cfg) - set(settings):
        raise ValueError(
            f"obsolete/unknown retrieval settings: {sorted(set(cfg) - set(settings))}"
        )
    settings.update(cfg)
    if settings["mode"] not in ("shortlist", "full", "coarse"):
        raise ValueError("retrieval.mode must be shortlist, full or coarse")
    for key in ("top_m", "fine_batch_size", "identity_batch_size", "coarse_batch_size"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"retrieval.{key} must be positive")
    return settings


def encode_gallery_identity(cache, model, gallery_ids, device, batch_size):
    """Refresh once for CURRENT P_id on every retrieval/mining call; retain on CPU."""
    batches = []
    indices = torch.tensor([cache.by_id[i] for i in gallery_ids])
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        mask = cache.mask[selected]
        columns = mask.any(0)
        mask = mask[:, columns]
        persons = (
            cache.persons[selected][:, columns].float().masked_fill(~mask[..., None], 0)
        )
        batches.append((model.encode_identity(persons.to(device)).cpu(), mask))
    return batches


def score_coarse(query, gallery_batches, model, device, batch_size):
    scores = []
    for identity, mask in gallery_batches:
        for start in range(0, len(identity), batch_size):
            scores.append(
                coarse_scores(
                    query,
                    identity[start : start + batch_size].to(device),
                    mask[start : start + batch_size].to(device),
                    model.matching,
                ).cpu()
            )
    return torch.cat(scores)


def score_fine(query, local_indices, gallery_ids, cache, model, device, batch_size):
    scores = []
    for start in range(0, len(local_indices), batch_size):
        local = local_indices[start : start + batch_size]
        indices = torch.tensor([cache.by_id[gallery_ids[i]] for i in local.tolist()])
        target = to_device(cache.load(indices), device)
        scores.append(
            model.score_target(
                query.repeat_candidates(len(local)), target, cache.patch_hw
            )["scores"].cpu()
        )
    return torch.cat(scores) if scores else torch.empty(0)


def rank_order(coarse, fine_indices, fine_scores, query_index):
    """Fine shortlist followed by its unchanged coarse tail, with canonical ties."""
    order = torch.argsort(coarse, descending=True, stable=True)
    order = order[order != query_index]
    selected = torch.zeros(len(coarse), dtype=torch.bool)
    selected[fine_indices] = True
    fine = fine_indices[torch.argsort(fine_scores, descending=True, stable=True)]
    return torch.cat((fine, order[~selected[order]])), order


@torch.inference_mode()
def retrieve_rankings(
    samples, cache, model, device, *, gallery_ids, description="retrieve", **retrieval
):
    cfg = retrieval_settings(retrieval)
    if not samples or len(gallery_ids) < 2 or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("retrieval needs queries and a unique gallery with >=2 images")
    gallery_by_id = {image_id: i for i, image_id in enumerate(gallery_ids)}
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    final, coarse_rows, shortlists = [], [], []
    try:
        gallery_batches = encode_gallery_identity(
            cache, model, gallery_ids, device, cfg["identity_batch_size"]
        )
        for sample in tqdm(samples, desc=description):
            inputs = to_device(query_inputs([sample], cache), device)
            query = model.encode_query(inputs["visual"], inputs["text"], cache.patch_hw)
            coarse = score_coarse(
                query, gallery_batches, model, device, cfg["coarse_batch_size"]
            )
            query_index = gallery_by_id[sample["query_image_id"]]
            order = torch.argsort(coarse, descending=True, stable=True)
            order = order[order != query_index]
            shortlists.append(order[: cfg["top_m"]])
            chosen = (
                (order if cfg["mode"] == "full" else order[: cfg["top_m"]])
                if cfg["mode"] != "coarse"
                else order[:0]
            )
            fine_scores = score_fine(
                query, chosen, gallery_ids, cache, model, device, cfg["fine_batch_size"]
            )
            result, coarse_order = rank_order(coarse, chosen, fine_scores, query_index)
            final.append(result.to(torch.int32))
            coarse_rows.append(coarse_order.to(torch.int32))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        return {
            "sample_ids": [s["sample_id"] for s in samples],
            "gallery_ids": list(gallery_ids),
            "rankings": torch.stack(final),
            "coarse_rankings": torch.stack(coarse_rows),
            "coarse_topm": torch.stack(shortlists).to(torch.int32),
            "runtime": {
                "retrieval_seconds": elapsed,
                "seconds_per_query": elapsed / len(samples),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "scoring_policy": cfg["mode"],
                "shortlist_approximation": cfg["mode"] == "shortlist",
            },
        }
    finally:
        model.train(was_training)


@torch.inference_mode()
def mine_hard_negatives(
    samples, cache, model, device, *, gallery_ids, retrieval, pool_size, excluded=None
):
    """Train gallery only; coarse candidate pool -> CURRENT fine scorer -> hard pool."""
    if gallery_ids != split_image_ids(cache.data, "train") or not {
        s["sample_id"] for s in samples
    } <= set(cache.data.splits["train"]):
        raise ValueError("negative mining may use only train queries and train gallery")
    cfg = retrieval_settings(retrieval)
    modes = model.training
    model.eval()
    excluded = excluded or {}
    pools = {}
    try:
        gallery = encode_gallery_identity(
            cache, model, gallery_ids, device, cfg["identity_batch_size"]
        )
        for sample in tqdm(samples, desc="mine coarse + fine train negatives"):
            inputs = to_device(query_inputs([sample], cache), device)
            query = model.encode_query(inputs["visual"], inputs["text"], cache.patch_hw)
            coarse = score_coarse(
                query, gallery, model, device, cfg["coarse_batch_size"]
            )
            forbidden = {
                sample["query_image_id"],
                *sample["positive_image_ids"],
                *excluded.get(sample["sample_id"], ()),
            }
            order = torch.argsort(coarse, descending=True, stable=True)
            order = torch.tensor(
                [i for i in order.tolist() if gallery_ids[i] not in forbidden]
            )[: max(pool_size, cfg["top_m"])]
            scores = score_fine(
                query, order, gallery_ids, cache, model, device, cfg["fine_batch_size"]
            )
            selected = order[torch.argsort(scores, descending=True, stable=True)][
                :pool_size
            ]
            pools[sample["sample_id"]] = [gallery_ids[i] for i in selected.tolist()]
        return pools
    finally:
        model.train(modes)
