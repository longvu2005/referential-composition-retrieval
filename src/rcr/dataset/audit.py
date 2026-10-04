"""Read-only checks for equivalent labels and detected identity coverage."""

from collections import defaultdict

from rcr.dataset.review import normalize_review_assignment
from rcr.dataset.rewrite import validate_review_output
from rcr.methods.common.data import split_image_ids, split_samples


def positive_conflicts(samples: list[dict]) -> list[dict]:
    """Find disagreements for identical conditions and ordered Subject identities.

    Call separately for each split. A query image has no negative vote in its
    own query. These are review candidates, never automatically corrected labels.
    """
    groups = defaultdict(list)
    for sample in samples:
        key = (
            sample["case_type"],
            tuple(
                (s["subject_id"], tuple(sorted(map(str, s["identity_ids"]))))
                for s in sample["subjects"]
            ),
            sample["final_change"],
        )
        groups[key].append(sample)
    conflicts = []
    for rows in groups.values():
        positives = [set(row["positive_image_ids"]) for row in rows]
        disputed = []
        for image_id in sorted(set().union(*positives)):
            votes = {
                image_id in positive
                for row, positive in zip(rows, positives, strict=True)
                if image_id != row["query_image_id"]
            }
            if len(votes) > 1:
                disputed.append(image_id)
        if disputed:
            conflicts.append(
                {
                    "case_type": rows[0]["case_type"],
                    "subjects": rows[0]["subjects"],
                    "final_change": rows[0]["final_change"],
                    "disputed_image_ids": disputed,
                    "samples": [
                        {
                            key: row[key]
                            for key in (
                                "sample_id",
                                "query_image_id",
                                "positive_image_ids",
                            )
                        }
                        for row in rows
                    ],
                }
            )
    return conflicts


def negative_exclusions(samples: list[dict]) -> dict[str, set[str]]:
    """Do not turn another equivalent query's positive into a training negative.

    Retain each query's reviewed positives. Disputed negative pairs are ignored;
    no positives are merged and no evaluation labels are modified.
    """
    excluded = {}
    for group in positive_conflicts(samples):
        for row in group["samples"]:
            excluded[row["sample_id"]] = (
                set(group["disputed_image_ids"])
                - set(row["positive_image_ids"])
                - {row["query_image_id"]}
            )
    return excluded


def audit_data(data) -> dict:
    """Check structural labels, disjoint image/identity splits, and conflicts."""
    from rcr.evaluation.evaluate import identity_positive_ids

    identities = {
        image_id: {str(b["identity_id"]) for b in boxes}
        for image_id, boxes in data.gt_head_boxes_by_image.items()
    }
    report, split_identities = {}, {}
    for split in ("train", "val", "test"):
        samples = split_samples(data, split)
        gallery = split_image_ids(data, split)
        split_identities[split] = set().union(
            *(identities.get(image_id, set()) for image_id in gallery)
        )
        cases = defaultdict(int)
        for sample in samples:
            cases[sample["case_type"]] += 1
            normalize_review_assignment(
                sample["sample_id"], sample["case_type"], sample["subjects"]
            )
            validate_review_output(
                sample["case_type"],
                {k: sample[k] for k in ("final_desc", "final_change")},
            )
            required = {
                str(identity)
                for subject in sample["subjects"]
                for identity in subject["identity_ids"]
            }
            if not required <= identities.get(sample["query_image_id"], set()):
                raise ValueError(f"{sample['sample_id']}: query lacks required IDs")
            positives = sample["positive_image_ids"]
            if not positives or len(positives) != len(set(positives)):
                raise ValueError(f"{sample['sample_id']}: empty/duplicate positives")
            if sample["target_image_id"] not in positives:
                raise ValueError(f"{sample['sample_id']}: missing seed positive")
            if not set(positives) <= set(
                identity_positive_ids(sample, gallery, identities)
            ):
                raise ValueError(f"{sample['sample_id']}: invalid Full Positive")
        conflicts = positive_conflicts(samples)
        report[split] = {
            "queries": len(samples),
            "gallery_images": len(gallery),
            "cases": dict(cases),
            "identity_count": len(split_identities[split]),
            "conflict_groups": len(conflicts),
            "conflict_queries": sum(len(g["samples"]) for g in conflicts),
            "excluded_negative_pairs": sum(
                len(ids) for ids in negative_exclusions(samples).values()
            ),
            "conflicts": conflicts,
        }
    overlap = {
        f"{a}/{b}": sorted(split_identities[a] & split_identities[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {"splits": report, "identity_overlap": overlap}


def audit_detection(data, cache) -> dict:
    """Read query/positive cache entries once; report ALL required ID coverage.

    Coverage measures cached GT alignment, not detector localization accuracy.
    It never changes a feature, detector box, label, or ranking.
    """
    cache.validate_gallery(data.gallery_ids)
    by_id = {image_id: i for i, image_id in enumerate(cache.image_ids)}
    detected = {}

    def identities(image_id):
        if image_id not in detected:
            item = cache._load_item(by_id[image_id])
            detected[image_id] = {
                str(identity)
                for identity in item.get("identity_ids", [])
                if identity is not None
            }
        return detected[image_id]

    report = {}
    for split in ("train", "val", "test"):
        counts = defaultdict(lambda: defaultdict(int))
        for sample in split_samples(data, split):
            query_ids = identities(sample["query_image_id"])
            subjects = [set(map(str, s["identity_ids"])) for s in sample["subjects"]]
            required = set().union(*subjects)
            values = {
                "queries": 1,
                "query_all_identities_detected": int(required <= query_ids),
                "subjects": len(subjects),
                "subjects_any_detected": sum(bool(s & query_ids) for s in subjects),
                "subjects_all_detected": sum(s <= query_ids for s in subjects),
                "positive_targets": len(sample["positive_image_ids"]),
                "positive_targets_all_detected": sum(
                    required <= identities(image_id)
                    for image_id in sample["positive_image_ids"]
                ),
            }
            for case in ("overall", sample["case_type"]):
                for key, value in values.items():
                    counts[case][key] += value
        report[split] = {case: dict(values) for case, values in counts.items()}
    return {"cache_id": cache.cache_id, "splits": report}
