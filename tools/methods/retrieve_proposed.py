"""Retrieve gallery images with the proposed RCR model."""

import argparse
from pathlib import Path

import torch
import yaml

from rcr.methods.common.data import load_rcr_data, split_samples
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import TextEncoder
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.retrieval import retrieve_rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/methods/proposed/retrieve.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    retrieval_cfg = cfg["retrieval"]
    device_name = retrieval_cfg["device"]
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

    data_cfg = cfg["data"]
    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, cfg["split"])
    cache = GalleryCache(data_cfg["cache"])
    cache.validate_gallery(data.gallery_ids)
    if "cache_id" in checkpoint and cache.cache_id != checkpoint["cache_id"]:
        raise ValueError("retrieval cache differs from the training cache")

    output = retrieve_rankings(
        samples,
        cache,
        tokenizer,
        text_encoder,
        model,
        device,
        top_m=retrieval_cfg["top_m"],
        fine_batch_size=retrieval_cfg["fine_batch_size"],
        identity_batch_size=retrieval_cfg["identity_batch_size"],
        coarse_batch_size=int(retrieval_cfg.get("coarse_batch_size", 512)),
        description=cfg["split"],
    )

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_dir / "rankings.pt")


if __name__ == "__main__":
    main()
