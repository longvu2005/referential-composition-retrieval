import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = Path(__file__).resolve().parent

PAIRS = ROOT / "dataset/data/raw/annotations/export_stage2.jsonl"
PAIR_DATA = ROOT / "dataset/data/raw/metadata/pair_data.json"
IMAGE_ROOT = ROOT / "dataset/data/raw/images"

MODEL = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(device).eval()

pairs = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]

metadata = json.loads(PAIR_DATA.read_text())
images = {x["image_id"]: x["url"] for x in metadata["images"]}


def image_path(image_id):
    relative = images[image_id].split("/PIPA/images/", 1)[1]
    return IMAGE_ROOT / relative


@torch.no_grad()
def count_people(image_id):
    image = Image.open(image_path(image_id)).convert("RGB")

    inputs = processor(
        images=image,
        text="person.",
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )[0]

    keep = nms(result["boxes"], result["scores"], 0.5)
    return len(keep)


# Detect mỗi ảnh đúng một lần.
image_ids = {
    image_id
    for pair in pairs
    for image_id in (pair["query_image_id"], pair["target_image_id"])
}

counts = {
    image_id: count_people(image_id)
    for image_id in tqdm(image_ids, desc="Detect people")
}

single_person_triplets = [
    pair
    for pair in pairs
    if counts[pair["query_image_id"]] == 1 and counts[pair["target_image_id"]] == 1
]

print(f"Total triplets: {len(pairs)}")
print(f"Both images have exactly 1 person: {len(single_person_triplets)}")
print(f"Ratio: {len(single_person_triplets) / len(pairs):.2%}")
