"""Build the two human-labeling handoff catalogs."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath

from rcr.common.data import SPLIT_NAMES
from rcr.common.io import index_unique
from rcr.dataset.review import normalize_review_assignment
from rcr.dataset.text import validate_review_output

CandidateRanker = Callable[[str, list[tuple[str, str]]], list[str]]


def _index_metadata(index_lines: Iterable[str]):
    boxes_by_image = defaultdict(list)
    images_by_identity = defaultdict(set)

    for line in index_lines:
        album, photo, x, y, w, h, identity_id, _ = line.split()
        image_id = f"{album}_{photo}"
        boxes_by_image[image_id].append(
            {
                "identity_id": identity_id,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            }
        )
        images_by_identity[identity_id].add(image_id)

    return boxes_by_image, images_by_identity


def merge_handoff_catalog(
    existing: Iterable[dict], prepared: Iterable[dict]
) -> list[dict]:
    existing = index_unique(existing, "sample_id", "existing handoff catalog")
    prepared = index_unique(prepared, "sample_id", "prepared handoff rows")
    return [prepared.get(key, row) for key, row in existing.items()] + [
        row for key, row in prepared.items() if key not in existing
    ]


def prepare_review_inputs(selected: list[dict]) -> list[dict]:
    """Attach image/identity context directly to accepted Stage 2 text."""
    selected_by_id = index_unique(selected, "sample_id", "selected")
    outputs = []

    for source in selected_by_id.values():
        sample_id = source["sample_id"]
        final_desc, final_change = validate_review_output(
            source["case_type"],
            {
                "final_desc": source["final_desc"],
                "final_change": source["final_change"],
            },
        )
        query_id, target_id = source["query_image_id"], source["target_image_id"]
        subjects = source["subjects"]

        subject_by_identity = {}
        for subject in subjects:
            for identity_id in subject["identity_ids"]:
                if identity_id in subject_by_identity:
                    raise ValueError(
                        f"{sample_id}: identity_id {identity_id} "
                        "is assigned to multiple subjects"
                    )
                subject_by_identity[identity_id] = subject["subject_id"]

        query_ids = {x["identity_id"] for x in source["query_boxes"]}
        target_ids = {x["identity_id"] for x in source["target_boxes"]}
        candidate_ids = sorted(query_ids & target_ids, key=int)
        missing = set(subject_by_identity) - set(candidate_ids)
        if missing:
            raise ValueError(
                f"{sample_id}: subject identities are not present "
                "in both query and seed target: " + ", ".join(sorted(missing))
            )

        outputs.append(
            {
                "sample_id": sample_id,
                "case_type": source["case_type"],
                "query_image_id": query_id,
                "query_image_url": f"/images/{source['query_image_path']}",
                "target_image_id": target_id,
                "target_image_url": f"/images/{source['target_image_path']}",
                "subjects": subjects,
                "candidate_identity_ids": candidate_ids,
                "query_boxes": source["query_boxes"],
                "target_boxes": source["target_boxes"],
                "final_desc": final_desc,
                "final_change": final_change,
            }
        )

    return outputs


def prepare_positive_set_inputs(
    selected: list[dict],
    reviewed: list[dict],
    gallery_images: list[dict],
    index_lines: Iterable[str],
    candidate_ranker: CandidateRanker | None = None,
) -> list[dict]:
    """Build all identity-compatible candidates within each query's image split."""
    selected_by_id = index_unique(selected, "sample_id", "selected")
    reviewed_by_id = index_unique(reviewed, "sample_id", "reviewed")
    image_by_id = index_unique(gallery_images, "image_id", "gallery_images")
    split_by_image = {
        image_id: PurePosixPath(image["path"]).parts[0]
        for image_id, image in image_by_id.items()
    }
    images_by_split = {split: set() for split in SPLIT_NAMES}
    for image_id, split in split_by_image.items():
        if split in images_by_split:
            images_by_split[split].add(image_id)

    boxes_by_image, images_by_identity = _index_metadata(index_lines)
    indexed_ids = set(boxes_by_image)
    missing = indexed_ids - set(image_by_id)
    if missing:
        raise ValueError(f"indexed image {min(missing)} is missing from gallery_images")

    outputs = []
    for review in reviewed_by_id.values():
        sample_id = review["sample_id"]
        source = selected_by_id[sample_id]
        query_id, seed_id = source["query_image_id"], source["target_image_id"]
        if query_id not in image_by_id or seed_id not in image_by_id:
            raise ValueError(
                f"{sample_id}: query and seed target must be registered gallery images"
            )
        query_split = split_by_image[query_id]
        if query_split not in images_by_split:
            raise ValueError(
                f"{sample_id}: query must belong to a train, val or test image split"
            )
        if split_by_image[seed_id] != query_split:
            raise ValueError(
                f"{sample_id}: seed target must belong to the same split as the query"
            )
        case_type = review.get("case_type", source["case_type"])
        case_type, subjects = normalize_review_assignment(
            sample_id, case_type, review.get("subjects")
        )
        validate_review_output(
            case_type,
            {
                "final_desc": review["final_desc"],
                "final_change": review["final_change"],
            },
        )

        subject_by_identity = {
            identity_id: subject["subject_id"]
            for subject in subjects
            for identity_id in subject["identity_ids"]
        }

        for identity_id in subject_by_identity:
            if not {query_id, seed_id} <= images_by_identity.get(identity_id, set()):
                raise ValueError(
                    f"{sample_id}: reviewed identity_id {identity_id} "
                    "must appear in both query and seed target"
                )

        candidate_ids = images_by_split[query_split].intersection(
            *(images_by_identity[x] for x in subject_by_identity)
        )
        candidate_ids.discard(query_id)
        if seed_id not in candidate_ids:
            raise ValueError(f"seed target {seed_id} is not identity-compatible")

        non_seed = sorted(candidate_ids - {seed_id})
        if candidate_ranker is not None:
            ranked = candidate_ranker(
                review["final_change"],
                [(x, image_by_id[x]["url"]) for x in non_seed],
            )
            if len(ranked) != len(non_seed) or set(ranked) != set(non_seed):
                raise ValueError(
                    f"{sample_id}: candidate reranker must return "
                    "every non-seed candidate exactly once"
                )
            non_seed = ranked

        candidates = [
            {
                "image_id": image_id,
                "image_url": image_by_id[image_id]["url"],
                "is_seed": image_id == seed_id,
                "subject_boxes": [
                    {"subject_id": subject_by_identity[x["identity_id"]], **x}
                    for x in boxes_by_image[image_id]
                    if x["identity_id"] in subject_by_identity
                ],
            }
            for image_id in [seed_id, *non_seed]
        ]

        outputs.append(
            {
                "sample_id": sample_id,
                "case_type": case_type,
                "query_image_id": query_id,
                "query_image_url": image_by_id[query_id]["url"],
                "seed_target_image_id": seed_id,
                "subjects": subjects,
                "final_desc": review["final_desc"],
                "final_change": review["final_change"],
                "candidates": candidates,
                "positive_image_ids": [seed_id],
            }
        )

    return outputs
