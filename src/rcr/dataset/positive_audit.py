"""Read-only Full Positive integrity and review-scope audit; no model scoring."""

from collections import Counter, defaultdict

from rcr.common.data import required_identity_ids, split_image_ids
from rcr.common.io import index_unique
from rcr.dataset.audit import positive_conflicts
from rcr.dataset.positives import validate_positive_set


def audit_full_positives(data, catalog=None, decisions=None):
    """Audit labels against all ID-compatible images in each query's gallery.

    A matching catalog plus submitted decision establishes candidate *scope*,
    not that every image was actually inspected or that the semantics are right.
    Without these artifacts, negative review coverage is unknown.
    """
    tasks = index_unique(catalog or [], "sample_id", "positive catalog")
    saved = index_unique(decisions or [], "sample_id", "positive decisions")
    registered_images = set(data.images_by_id)
    identities = {
        i: {str(r["identity_id"]) for r in rows}
        for i, rows in data.gt_head_boxes_by_image.items()
    }
    by_identity = defaultdict(set)
    for image_id, ids in identities.items():
        for identity in ids:
            by_identity[identity].add(image_id)
    splits = {}
    for split, sample_ids in data.splits.items():
        gallery = split_image_ids(data, split)
        positions = {image_id: i for i, image_id in enumerate(gallery)}
        gallery_set = set(gallery)

        def ordered(ids, positions=positions):
            return sorted(ids, key=positions.__getitem__)

        rows = []
        projected = [
            {
                **data.samples_by_id[i],
                "positive_image_ids": [
                    p
                    for p in data.samples_by_id[i]["positive_image_ids"]
                    if p in gallery_set
                ],
            }
            for i in sample_ids
        ]
        conflicts = positive_conflicts(projected)
        disputed = defaultdict(set)
        for group in conflicts:
            for sample in group["samples"]:
                disputed[sample["sample_id"]].update(group["disputed_image_ids"])
        for sample_id in sample_ids:
            sample = data.samples_by_id[sample_id]
            query, seed = sample["query_image_id"], sample["target_image_id"]
            required = set(map(str, required_identity_ids(sample)))
            eligible = (
                set.intersection(*(by_identity[i] for i in required))
                if required
                else set()
            ) & gallery_set - {query}
            raw = sample["positive_image_ids"]
            positive_ids = set(raw)
            full = positive_ids & gallery_set
            issues = []
            if not required:
                issues.append("empty_required_identities")
            if query not in gallery_set:
                issues.append("query_outside_split")
            if not required <= identities.get(query, set()):
                issues.append("query_missing_required_identities")
            if seed not in gallery_set:
                issues.append("seed_outside_split")
            if seed not in full:
                issues.append("missing_seed_positive")
            if not full:
                issues.append("empty_split_positive_set")
            if len(raw) != len(positive_ids):
                issues.append("duplicate_positive_ids")
            if query in positive_ids:
                issues.append("self_positive")
            if positive_ids - registered_images:
                issues.append("unregistered_positive_ids")
            invalid = {i for i in raw if not required <= identities.get(i, set())}
            if invalid:
                issues.append("positive_missing_required_identities")

            task, decision = tasks.get(sample_id), saved.get(sample_id)
            source_issues = []
            scope = set()
            if task is not None:
                context = {
                    "query_image_id": query,
                    "seed_target_image_id": seed,
                    **{
                        key: sample[key]
                        for key in (
                            "case_type",
                            "subjects",
                            "final_desc",
                            "final_change",
                        )
                    },
                }
                if any(task.get(key) != value for key, value in context.items()):
                    source_issues.append("stale_catalog_context")
                try:
                    checked = validate_positive_set(
                        task,
                        decision
                        if decision is not None
                        else {"sample_id": sample_id, "positive_image_ids": [seed]},
                    )
                    if decision is not None and (
                        set(checked["positive_image_ids"]) & gallery_set != full
                    ):
                        source_issues.append("decision_differs_from_final")
                except (ValueError, KeyError) as error:
                    source_issues.append(f"invalid_decision: {error}")
            if source_issues:
                status = "stale"
            elif task is None or decision is None:
                status = "unknown"
            else:
                scope = {c["image_id"] for c in task["candidates"]} & eligible
                status = "complete_catalog" if eligible <= scope else "partial_catalog"
            unconfirmed = eligible - full - scope
            rows.append(
                {
                    "sample_id": sample_id,
                    "case_type": sample["case_type"],
                    "has_catalog": task is not None,
                    "has_decision": decision is not None,
                    "annotation_issues": issues,
                    "review_source_issues": source_issues,
                    "scope_status": status,
                    "id_positive_image_ids": ordered(eligible),
                    "full_positive_image_ids": ordered(full),
                    "out_of_split_positive_image_ids": sorted(
                        positive_ids - gallery_set
                    ),
                    "invalid_identity_positive_image_ids": sorted(invalid),
                    "missing_catalog_image_ids": ordered(eligible - scope),
                    "unconfirmed_negative_image_ids": ordered(unconfirmed),
                    "catalog_negative_image_ids": ordered(scope - full),
                    "disputed_image_ids": ordered(
                        disputed[sample_id] & gallery_set - {query}
                    ),
                }
            )
        splits[split] = {
            "summary": {
                "queries": len(rows),
                "annotation_issue_queries": sum(
                    bool(r["annotation_issues"]) for r in rows
                ),
                "review_source_issue_queries": sum(
                    bool(r["review_source_issues"]) for r in rows
                ),
                "scope_status_counts": dict(Counter(r["scope_status"] for r in rows)),
                "id_positive_pairs": sum(len(r["id_positive_image_ids"]) for r in rows),
                "full_positive_pairs": sum(
                    len(r["full_positive_image_ids"]) for r in rows
                ),
                "unconfirmed_negative_pairs": sum(
                    len(r["unconfirmed_negative_image_ids"]) for r in rows
                ),
                "conflict_groups": len(conflicts),
            },
            "conflicts": conflicts,
            "queries": rows,
        }
    return {
        "catalog_supplied": catalog is not None,
        "decisions_supplied": decisions is not None,
        "semantic_completeness": "not_established_by_code",
        "policy": (
            "read-only; ID-compatible non-Full images are review candidates, "
            "not inferred positives"
        ),
        "splits": splits,
    }


def full_positive_review_queue(data, audit):
    """New, split-scoped UI catalog for unresolved queries; keep saved labels."""
    tasks = []
    for split, report in audit["splits"].items():
        for row in report["queries"]:
            if row["annotation_issues"]:
                continue  # Repair structural problems before asking the positive UI.
            if (
                row["scope_status"] == "complete_catalog"
                and not row["disputed_image_ids"]
            ):
                continue
            sample = data.samples_by_id[row["sample_id"]]
            seed = sample["target_image_id"]
            subject_by_id = {
                str(identity): s["subject_id"]
                for s in sample["subjects"]
                for identity in s["identity_ids"]
            }
            candidates = [seed, *[i for i in row["id_positive_image_ids"] if i != seed]]
            tasks.append(
                {
                    "sample_id": sample["sample_id"],
                    "case_type": sample["case_type"],
                    "query_image_id": sample["query_image_id"],
                    "query_image_url": "/images/"
                    + data.images_by_id[sample["query_image_id"]]["path"],
                    "seed_target_image_id": seed,
                    "subjects": sample["subjects"],
                    "final_desc": sample["final_desc"],
                    "final_change": sample["final_change"],
                    "positive_image_ids": list(row["full_positive_image_ids"]),
                    "audit_split": split,
                    "candidates": [
                        {
                            "image_id": i,
                            "image_url": "/images/" + data.images_by_id[i]["path"],
                            "is_seed": i == seed,
                            "subject_boxes": [
                                {
                                    **head,
                                    "subject_id": subject_by_id[
                                        str(head["identity_id"])
                                    ],
                                }
                                for head in data.gt_head_boxes_by_image.get(i, [])
                                if str(head["identity_id"]) in subject_by_id
                            ],
                        }
                        for i in candidates
                    ],
                }
            )
    return tasks
