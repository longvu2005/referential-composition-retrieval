"""Label-free model inputs and explicitly separate, train-only supervision."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch

from rcr.proposed.nn.encoders import parse_subjects


def to_device(value, device):
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value.to(device, non_blocking=True)


def query_inputs(samples, cache):
    """Only Iq, selection and full change; safe with GT fields entirely removed."""
    indices = torch.tensor([cache.by_id[s["query_image_id"]] for s in samples])
    return {"visual": cache.load(indices), "text": cache.text.batch(samples)}


def build_supervision(samples, candidates, query, target, labels, train_identities):
    b, c = len(samples), len(candidates[0])
    q, t = query["person_mask"].shape[1], target["person_mask"].shape[1]
    grounding = torch.full((b, q), -1, dtype=torch.long)
    query_ids = torch.full((b, q), -1, dtype=torch.long)
    target_ids = torch.full((b, c, t), -1, dtype=torch.long)
    matching = torch.zeros(b, c, q, 1 + t, dtype=torch.bool)
    match_mask = torch.zeros(b, c, q, dtype=torch.bool)
    query_unique = torch.zeros(b, q, dtype=torch.bool)
    target_unique = torch.zeros(b, c, t, dtype=torch.bool)
    seen = set()

    def identity_row(image_id, out, unique):
        for i, identity in enumerate(labels[image_id]["identity_ids"]):
            if identity is None or str(identity) not in train_identities:
                continue
            out[i] = train_identities[str(identity)]
            crop = image_id, i
            if crop not in seen:
                unique[i] = True
                seen.add(crop)

    for n, sample in enumerate(samples):
        image_id = sample["query_image_id"]
        identity_row(image_id, query_ids[n], query_unique[n])
        role_ids, _, _ = parse_subjects(sample["final_desc"], sample["final_change"])
        annotations = {
            int(x["subject_id"]): set(map(str, x["identity_ids"]))
            for x in sample["subjects"]
        }
        if set(annotations) != set(role_ids):
            raise ValueError("text/annotation Subject IDs disagree")
        for i, identity in enumerate(labels[image_id]["identity_ids"]):
            if identity is None or str(identity) not in train_identities:
                continue
            roles = [
                j + 1
                for j, sid in enumerate(role_ids)
                if str(identity) in annotations[sid]
            ]
            # Conflicting identity annotations cannot supervise categorical roles.
            if len(roles) <= 1:
                grounding[n, i] = roles[0] if roles else 0
        for j, target_id in enumerate(candidates[n]):
            identity_row(target_id, target_ids[n, j], target_unique[n, j])
            target_label = labels[target_id]
            for i, identity in enumerate(labels[image_id]["identity_ids"]):
                if identity is None or grounding[n, i] <= 0:
                    continue
                positives = [
                    k
                    for k, tid in enumerate(target_label["identity_ids"])
                    if tid is not None and str(tid) == str(identity)
                ]
                if positives:
                    matching[n, j, i, torch.tensor(positives) + 1] = True
                    match_mask[n, j, i] = True
                elif all(tid is not None for tid in target_label["identity_ids"]) and (
                    str(identity) in target_label["gt_ids"] or target_label["complete"]
                ):
                    # Confirmed detector miss or explicitly complete annotation.
                    # Unmatched/unknown detections cannot prove absence.
                    matching[n, j, i, 0] = True
                    match_mask[n, j, i] = True
    positive_mask = torch.tensor(
        [
            [image_id in set(sample["positive_image_ids"]) for image_id in row]
            for sample, row in zip(samples, candidates, strict=True)
        ]
    )
    return {
        "grounding": grounding,
        "query_ids": query_ids,
        "target_ids": target_ids,
        "query_unique": query_unique,
        "target_unique": target_unique,
        "match_positive": matching,
        "match_mask": match_mask,
        "positive_mask": positive_mask,
        "candidate_mask": torch.ones_like(positive_mask),
    }


def build_batch(samples, candidates, cache, labels, train_identities):
    inputs = query_inputs(samples, cache)
    b, c = len(samples), len(candidates[0])
    indices = torch.tensor([cache.by_id[i] for row in candidates for i in row])
    target = cache.load(indices)
    supervision = build_supervision(
        samples, candidates, inputs["visual"], target, labels, train_identities
    )
    return {
        "inputs": {
            "query": inputs,
            "target": {k: v.reshape(b, c, *v.shape[1:]) for k, v in target.items()},
        },
        "supervision": supervision,
    }


@contextmanager
def prefetch_batches(jobs, prepare, depth=0):
    """At most one CPU batch ahead; RNG/job generation stays in the caller."""
    if depth not in (0, 1):
        raise ValueError("cache.prefetch_batches must be 0 or 1")
    if not depth:
        yield map(prepare, jobs)
        return
    with ThreadPoolExecutor(max_workers=1) as executor:

        def batches():
            iterator = iter(jobs)
            first = next(iterator, None)
            if first is None:
                return
            pending = executor.submit(prepare, first)
            for job in iterator:
                batch = pending.result()
                pending = executor.submit(prepare, job)
                yield batch
            yield pending.result()

        iterator = batches()
        try:
            yield iterator
        finally:
            iterator.close()
