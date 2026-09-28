"""Build the final RCR dataset from completed handoffs."""

from collections import defaultdict
from collections.abc import Iterable

from rcr.dataset.gallery import indexed_image_ids
from rcr.dataset.review import normalize_review_assignment
from rcr.dataset.rewrite import validate_review_output

PAIR_SPLITS = frozenset({"TRAIN", "VAL", "TEST"})


def pair_split_index(pair_data: dict) -> dict[tuple[str, str], str]:
    """Index ordered query/target pairs by their authoritative PIPA split."""

    output: dict[tuple[str, str], str] = {}
    for pair in pair_data["pairs"]:
        query_id, target_id = pair["query_image_id"], pair["target_image_id"]
        split = pair["split"].upper()
        if split not in PAIR_SPLITS:
            raise ValueError(f"pair_data has invalid split {split!r}")
        key = (query_id, target_id)
        if key in output:
            raise ValueError(
                "pair_data contains duplicate ordered pair "
                f"{query_id!r} -> {target_id!r}"
            )
        output[key] = split

    return output


def _index_unique(records: Iterable[dict], label: str) -> dict[str, dict]:
    output = {}
    for row in records:
        sample_id = row["sample_id"]
        if sample_id in output:
            raise ValueError(f"{label}: duplicate sample_id {sample_id}")
        output[sample_id] = row
    return output


def build_final_dataset(
    selected: list[dict],
    reviewed: list[dict],
    positive_sets: list[dict],
    gallery_images: list[dict],
    index_lines: Iterable[str],
    version: str,
    pair_data: dict | None = None,
    allow_partial: bool = False,
) -> dict:
    """Build the final dataset, optionally using only completed handoffs."""

    selected_by_id = _index_unique(selected, "selected")
    reviewed_by_id = _index_unique(reviewed, "reviewed")
    positives_by_id = _index_unique(positive_sets, "positive_sets")
    expected_ids = set(selected_by_id)
    pair_splits = pair_split_index(pair_data) if pair_data is not None else None

    if allow_partial:
        selected = [
            row
            for row in selected
            if row["sample_id"] in reviewed_by_id
            and row["sample_id"] in positives_by_id
        ]
    else:
        if set(reviewed_by_id) != expected_ids:
            raise ValueError("reviewed samples do not match selected samples")
        if set(positives_by_id) != expected_ids:
            raise ValueError("positive sets do not match selected samples")

    index_lines = list(index_lines)
    index_ids = indexed_image_ids(index_lines)
    gallery_by_id = {x["image_id"]: x for x in gallery_images}
    if len(gallery_by_id) != len(gallery_images):
        raise ValueError("gallery_images contains duplicate image_id")
    if set(gallery_by_id) != set(index_ids):
        raise ValueError("gallery_images must match the complete indexed gallery")

    images = [
        {"image_id": image_id, "path": gallery_by_id[image_id]["path"]}
        for image_id in index_ids
    ]
    identities_by_image = defaultdict(set)
    head_boxes = []

    for line in index_lines:
        album, photo, x, y, w, h, identity_id, _ = line.split()
        image_id = f"{album}_{photo}"
        identities_by_image[image_id].add(identity_id)
        head_boxes.append(
            {
                "box_id": f"{image_id}::pid{identity_id}",
                "image_id": image_id,
                "identity_id": identity_id,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            }
        )

    samples = []
    splits = {"train": [], "val": [], "test": []}
    gallery_ids = set(index_ids)

    for source in selected:
        sample_id = source["sample_id"]
        review = reviewed_by_id[sample_id]
        positive_ids = positives_by_id[sample_id]["positive_image_ids"]
        query_id = source["query_image_id"]
        target_id = source["target_image_id"]

        if not positive_ids or len(positive_ids) != len(set(positive_ids)):
            raise ValueError(
                f"{sample_id}: positive_image_ids must be non-empty and unique"
            )
        if target_id not in positive_ids:
            raise ValueError(f"{sample_id}: missing seed positive")
        if query_id in positive_ids:
            raise ValueError(f"{sample_id}: query image is positive")

        case_type, subjects = normalize_review_assignment(
            sample_id,
            review.get("case_type", source["case_type"]),
            review.get("subjects"),
        )
        final_desc, final_change = validate_review_output(
            case_type,
            {
                "final_desc": review["final_desc"],
                "final_change": review["final_change"],
            },
        )
        required_ids = {
            identity_id
            for subject in subjects
            for identity_id in subject["identity_ids"]
        }

        if not required_ids <= identities_by_image[query_id]:
            raise ValueError(
                f"{sample_id}: reviewed identities must appear in the query image"
            )
        for positive_id in positive_ids:
            if (
                positive_id not in gallery_ids
                or not required_ids <= identities_by_image[positive_id]
            ):
                raise ValueError(f"{sample_id}: invalid positive {positive_id}")

        samples.append(
            {
                "sample_id": sample_id,
                "case_type": case_type,
                "query_image_id": query_id,
                "target_image_id": target_id,
                "positive_image_ids": positive_ids,
                "subjects": subjects,
                "final_desc": final_desc,
                "final_change": final_change,
                "final_instruction": f"{final_desc}; {final_change}.",
            }
        )
        if pair_splits is None:
            split = source["split"].upper()
            if split not in PAIR_SPLITS:
                raise ValueError(f"{sample_id}: invalid split {split!r}")
        else:
            key = (query_id, target_id)
            try:
                split = pair_splits[key]
            except KeyError as exc:
                raise ValueError(
                    f"{sample_id}: pair is missing from pair_data "
                    f"({query_id!r}, {target_id!r})"
                ) from exc
        splits[split.lower()].append(sample_id)

    return {
        "samples": samples,
        "images": images,
        "gallery": [{"image_id": x["image_id"]} for x in images],
        "head_boxes": head_boxes,
        "splits": splits,
        "manifest": {
            "version": version,
            "num_samples": len(samples),
            "num_images": len(images),
            "num_head_boxes": len(head_boxes),
            "splits": {name: len(ids) for name, ids in splits.items()},
            "partial": allow_partial,
        },
    }
