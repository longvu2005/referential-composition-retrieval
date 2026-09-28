"""Retrieve gallery images with the proposed RCR model."""

import argparse
import re
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from rcr.methods.common.data import load_rcr_data, sample_selection_texts, split_samples
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.coarse import coarse_scores
from rcr.methods.proposed.composition import composed_query_mask, reference_key_bias
from rcr.methods.proposed.encoders import TextEncoder
from rcr.methods.proposed.model import RCRModel


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/methods/proposed/retrieve.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_name = cfg["retrieval"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_cfg = checkpoint["config"]
    model_cfg = train_cfg["model"]
    dim = checkpoint["dim"]

    from transformers import AutoModel, AutoTokenizer

    tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_backbone = AutoModel.from_pretrained(model_cfg["text_model"])
    text_backbone.resize_token_embeddings(len(tokenizer))
    text_encoder = TextEncoder(text_backbone, dim).to(device)
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    text_encoder.eval()

    max_subjects = checkpoint["model"]["composition.role.weight"].shape[0]
    model = RCRModel(
        dim=dim,
        identity_dim=model_cfg["identity_dim"],
        num_heads=model_cfg["num_heads"],
        max_subjects=max_subjects,
        mlp_ratio=model_cfg["mlp_ratio"],
        geo_dim=model_cfg["geo_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, cfg["split"])
    cache = GalleryCache(data_cfg["cache"])
    by_id = {image_id: i for i, image_id in enumerate(cache.image_ids)}

    gallery_identity = []
    identity_batch = cfg["retrieval"]["identity_batch_size"]
    for start in range(0, len(cache.image_ids), identity_batch):
        persons = cache.persons[start : start + identity_batch].to(device)
        gallery_identity.append(model.identity_head(persons))
    gallery_identity = torch.cat(gallery_identity)
    gallery_mask = cache.mask.to(device)

    sample_ids = []
    rankings = []
    coarse_topm = []
    top_m = cfg["retrieval"]["top_m"]
    fine_batch = cfg["retrieval"]["fine_batch_size"]

    for sample in tqdm(samples, desc=cfg["split"]):
        query_index = by_id[sample["query_image_id"]]
        query_idx = torch.tensor([query_index])
        q_scene, q_persons, q_boxes, _, q_mask = cache.load(query_idx)
        q_scene = q_scene.to(device)
        q_persons = q_persons.to(device)
        q_boxes = q_boxes.to(device)
        q_mask = q_mask.to(device)

        selection = tokenizer(
            sample_selection_texts(sample),
            padding=True,
            return_tensors="pt",
        )
        selection, selection_mask = text_encoder(
            selection["input_ids"].to(device),
            selection["attention_mask"].to(device),
        )
        selection = selection.unsqueeze(0)
        selection_mask = selection_mask.unsqueeze(0)

        change_text = sample["final_change"]
        for subject in sample["subjects"]:
            subject_id = int(subject["subject_id"])
            change_text = re.sub(
                rf"\bSubject\s+{subject_id}\b",
                f"[S{subject_id}]",
                change_text,
            )

        encoded = tokenizer(change_text, return_tensors="pt")
        change_ids = encoded["input_ids"].to(device)
        change, change_mask = text_encoder(
            change_ids,
            encoded["attention_mask"].to(device),
        )

        subject_pos = torch.empty(
            1, len(sample["subjects"]), dtype=torch.long, device=device
        )
        subject_token_mask = torch.zeros(
            1,
            len(sample["subjects"]),
            change_ids.shape[1],
            dtype=torch.bool,
            device=device,
        )
        for j, subject in enumerate(sample["subjects"]):
            marker = f"[S{int(subject['subject_id'])}]"
            marker_id = tokenizer.convert_tokens_to_ids(marker)
            pos = (change_ids[0] == marker_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                raise ValueError(f"{marker} must be one tokenizer token and occur")
            subject_pos[0, j] = pos[0]
            subject_token_mask[0, j, pos] = True

        logits = model.grounding(
            q_scene,
            q_persons,
            q_boxes,
            selection,
            cache.patch_hw,
            selection_mask,
        )
        query_identity = model.identity_head(q_persons)
        composition_logits = logits.masked_fill(~q_mask[:, None], -torch.inf)

        coarse = coarse_scores(
            query_identity[0],
            logits[0],
            gallery_identity,
            gallery_mask,
            query_mask=q_mask[0],
        )
        coarse[query_index] = -torch.inf
        coarse_order = torch.argsort(coarse, descending=True)
        coarse_order = coarse_order[coarse_order != query_index]
        top_indices = coarse_order[: min(top_m, len(coarse_order))]

        query = model.composition(
            change,
            query_identity,
            composition_logits,
            subject_pos,
            change_mask,
            subject_token_mask=subject_token_mask,
            subject_ids=torch.tensor(
                [[int(x["subject_id"]) for x in sample["subjects"]]], device=device
            ),
        )
        query_mask = composed_query_mask(change, composition_logits, change_mask)
        prior = reference_key_bias(change, composition_logits, change_mask)

        fine_scores = []
        for start in range(0, len(top_indices), fine_batch):
            indices = top_indices[start : start + fine_batch].cpu()
            scene, persons, boxes, _, target_mask = cache.load(indices)
            scene = scene.to(device)
            persons = persons.to(device)
            boxes = boxes.to(device)
            target_mask = target_mask.to(device)

            n = len(indices)
            score = model.score_target(
                query.expand(n, -1, -1),
                query_mask.expand(n, -1),
                prior.expand(n, -1),
                scene,
                persons,
                boxes,
                cache.patch_hw,
                target_mask,
            )
            fine_scores.append(score)

        fine_scores = torch.cat(fine_scores) if fine_scores else coarse.new_empty(0)
        fine_order = top_indices[torch.argsort(fine_scores, descending=True)]
        rest = coarse_order[len(top_indices) :]
        final_order = torch.cat((fine_order, rest))

        sample_ids.append(sample["sample_id"])
        rankings.append(final_order.to(torch.int32).cpu())
        coarse_topm.append(top_indices.to(torch.int32).cpu())

    output = Path(cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_ids": sample_ids,
            "gallery_ids": cache.image_ids,
            "rankings": torch.stack(rankings),
            "coarse_topm": torch.stack(coarse_topm),
        },
        output / "rankings.pt",
    )


if __name__ == "__main__":
    main()
