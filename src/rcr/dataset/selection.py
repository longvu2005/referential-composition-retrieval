"""Select approved Stage 2 rows and check their source pair metadata."""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from rcr.dataset.review import normalize_review_assignment
from rcr.dataset.text import SUBJECT_RE, validate_review_output

STAGE2_FIELDS = {
    "sample_id",
    "split",
    "annotator_email",
    "query_image_id",
    "query_image_path",
    "query_boxes",
    "target_image_id",
    "target_image_path",
    "target_boxes",
    "case_type",
    "subjects",
    "final_desc",
    "final_change",
    "final_instruction",
}
SPLITS = {"TRAIN", "VAL", "TEST"}
BOX_FIELDS = {"identity_id", "x", "y", "width", "height"}


def _box_index(boxes: object, sample_id: str, side: str) -> dict[str, tuple]:
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(f"{sample_id}: {side}_boxes must be non-empty")
    output = {}
    for box in boxes:
        if not isinstance(box, dict) or set(box) != BOX_FIELDS:
            raise ValueError(f"{sample_id}: invalid {side} box fields")
        identity_id = box["identity_id"]
        if not isinstance(identity_id, str) or not identity_id:
            raise ValueError(f"{sample_id}: invalid {side} box identity")
        if identity_id in output:
            raise ValueError(f"{sample_id}: duplicate {side} box identity")
        x, y, width, height = (box[k] for k in ("x", "y", "width", "height"))
        if any(type(v) not in (int, float) for v in (x, y, width, height)):
            raise ValueError(f"{sample_id}: invalid {side} box coordinates")
        if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0):
            raise ValueError(f"{sample_id}: invalid {side} box bounds")
        if x + width > 1 + 1e-8 or y + height > 1 + 1e-8:
            raise ValueError(f"{sample_id}: invalid {side} box bounds")
        output[identity_id] = (x, y, width, height)
    return output


def select_samples(
    records: Iterable[dict], pair_data: dict | None = None
) -> list[dict]:
    """Keep the source order; check IDs, final text and source boxes once."""

    selected = list(records)
    pair_by_images = {}
    source_boxes = defaultdict(dict)
    if pair_data is not None:
        for pair in pair_data["pairs"]:
            key = (pair["query_image_id"], pair["target_image_id"])
            if key in pair_by_images:
                raise ValueError(f"duplicate pair in pair_data: {key}")
            pair_by_images[key] = pair
        for box in pair_data["boxes"]:
            image_boxes = source_boxes[box["image_id"]]
            if box["label"] in image_boxes:
                raise ValueError(f"duplicate source box in {box['image_id']}")
            image_boxes[box["label"]] = tuple(
                box[k] for k in ("x", "y", "width", "height")
            )

    seen_ids = set()
    seen_pairs = set()
    for row in selected:
        if set(row) != STAGE2_FIELDS:
            difference = sorted(set(row) ^ STAGE2_FIELDS)
            raise ValueError(f"Stage 2 fields differ: {difference}")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("invalid sample_id")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"{sample_id}: invalid split")
        key = (row["query_image_id"], row["target_image_id"])
        if sample_id in seen_ids or key in seen_pairs:
            raise ValueError(f"{sample_id}: duplicate sample or ordered pair")
        seen_ids.add(sample_id)
        seen_pairs.add(key)
        if key[0] == key[1]:
            raise ValueError(f"{sample_id}: query equals seed target")
        if not isinstance(row["annotator_email"], str) or not row["annotator_email"]:
            raise ValueError(f"{sample_id}: missing annotator_email")

        for side in ("query", "target"):
            image_id, path = row[f"{side}_image_id"], row[f"{side}_image_path"]
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"{sample_id}: invalid {side}_image_id")
            if not isinstance(path, str) or Path(path).is_absolute():
                raise ValueError(f"{sample_id}: invalid {side}_image_path")
            if ".." in Path(path).parts or Path(path).stem != image_id:
                raise ValueError(f"{sample_id}: invalid {side}_image_path")

        query_boxes = _box_index(row["query_boxes"], sample_id, "query")
        target_boxes = _box_index(row["target_boxes"], sample_id, "target")
        if query_boxes.keys() != target_boxes.keys():
            raise ValueError(f"{sample_id}: query/target identity sets differ")

        case_type, subjects = normalize_review_assignment(
            sample_id, row["case_type"], row["subjects"]
        )
        if case_type != row["case_type"] or subjects != row["subjects"]:
            raise ValueError(f"{sample_id}: non-canonical subject assignment")
        required_ids = {identity for s in subjects for identity in s["identity_ids"]}
        if not required_ids <= query_boxes.keys():
            raise ValueError(f"{sample_id}: subject identity missing from boxes")

        # Preserve strict validation: only attach context to the error, never
        # silently alter, skip, or accept an invalid annotation.
        try:
            validate_review_output(
                case_type,
                {"final_desc": row["final_desc"], "final_change": row["final_change"]},
            )
        except ValueError as exc:
            context = f"{sample_id} [{case_type}]: {exc}"
            if "final_change must mention exactly" in str(exc):
                expected = [subject["subject_id"] for subject in subjects]
                found = sorted(
                    {int(s) for s in SUBJECT_RE.findall(row["final_change"])}
                )
                context += (
                    f"\n  Expected Subjects: {expected}"
                    f"\n  Found Subjects: {found}"
                    f"\n  final_change: {row['final_change']!r}"
                )
            raise ValueError(context) from exc

        instruction = f"{row['final_desc']}; {row['final_change']}."
        if row["final_instruction"] != instruction:
            raise ValueError(f"{sample_id}: final_instruction mismatch")

        if pair_data is not None:
            pair = pair_by_images.get(key)
            if pair is None or pair["split"] != split or pair["pair_id"] != sample_id:
                raise ValueError(f"{sample_id}: pair_data mismatch")
            common = source_boxes[key[0]].keys() & source_boxes[key[1]].keys()
            if query_boxes.keys() != common:
                raise ValueError(f"{sample_id}: incomplete shared box identities")
            for image_id, boxes in zip(key, (query_boxes, target_boxes), strict=True):
                for identity_id, coords in boxes.items():
                    if source_boxes[image_id][identity_id] != coords:
                        raise ValueError(f"{sample_id}: incorrect box coordinates")

    return selected
