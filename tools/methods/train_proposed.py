"""Train the proposed RCR model from cached visual features."""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from rcr.methods.common.data import load_rcr_data, split_image_ids, split_samples
from rcr.methods.proposed.batch import build_batch
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.encoders import TextEncoder
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.sampling import sample_candidates
from rcr.methods.proposed.training import compute_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/methods/proposed/train.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    optim_cfg = cfg["optimizer"]
    loss_cfg = cfg["loss"]
    output_cfg = cfg["output"]

    from transformers import AutoModel, AutoTokenizer

    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    device_name = train_cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    data = load_rcr_data(data_cfg["final_dir"], data_cfg["image_root"])
    samples = split_samples(data, "train")
    cache = GalleryCache(data_cfg["cache"])
    negative_pool = train_cfg.get("negative_pool", "full_gallery")
    if negative_pool == "train_gallery":
        candidate_ids = split_image_ids(data, "train")
    elif negative_pool == "full_gallery":
        candidate_ids = cache.image_ids
    else:
        raise ValueError("negative_pool must be full_gallery or train_gallery")
    print(f"Training negative pool: {negative_pool} ({len(candidate_ids)} images)")

    first = torch.load(
        Path(data_cfg["cache"]) / "features" / "0.pt",
        map_location="cpu",
        weights_only=True,
    )
    dim = first["scene"].shape[-1]
    if dim % model_cfg["num_heads"]:
        raise ValueError("feature dim must be divisible by num_heads")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["[S1]", "[S2]"]})
    text_backbone = AutoModel.from_pretrained(model_cfg["text_model"])
    text_backbone.resize_token_embeddings(len(tokenizer))
    text_encoder = TextEncoder(text_backbone, dim).to(device)

    max_subjects = max(len(sample["subjects"]) for sample in samples)
    model = RCRModel(
        dim=dim,
        identity_dim=model_cfg["identity_dim"],
        num_heads=model_cfg["num_heads"],
        max_subjects=max_subjects,
        mlp_ratio=model_cfg["mlp_ratio"],
        geo_dim=model_cfg["geo_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": optim_cfg["lr"]},
            {"params": text_encoder.parameters(), "lr": optim_cfg["text_lr"]},
        ],
        weight_decay=optim_cfg["weight_decay"],
    )

    groups: dict[int, list[dict]] = {}
    for sample in samples:
        groups.setdefault(len(sample["subjects"]), []).append(sample)

    output = Path(output_cfg["dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "config.yaml")
    tokenizer.save_pretrained(output / "tokenizer")

    for epoch in range(train_cfg["epochs"]):
        generator = torch.Generator().manual_seed(seed + epoch)
        batches = []
        for rows in groups.values():
            order = torch.randperm(len(rows), generator=generator).tolist()
            rows = [rows[i] for i in order]
            batches.extend(
                rows[i : i + train_cfg["batch_size"]]
                for i in range(0, len(rows), train_cfg["batch_size"])
            )
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in order]

        model.train()
        text_encoder.train()
        totals = {
            "loss": 0.0,
            "grounding": 0.0,
            "identity": 0.0,
            "retrieval": 0.0,
        }

        progress = tqdm(
            batches,
            desc=f"epoch {epoch + 1}/{train_cfg['epochs']}",
        )
        for rows in progress:
            candidates = sample_candidates(
                rows,
                candidate_ids,
                train_cfg["candidates"],
                generator,
            )

            optimizer.zero_grad(set_to_none=True)
            batch = build_batch(
                rows,
                candidates,
                cache,
                tokenizer,
                text_encoder,
                device,
            )
            loss, parts = compute_loss(
                model,
                batch,
                cache.patch_hw,
                grounding_weight=loss_cfg["grounding_weight"],
                identity_weight=loss_cfg["identity_weight"],
                retrieval_weight=loss_cfg["retrieval_weight"],
                identity_temperature=loss_cfg["identity_temperature"],
            )
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            for name, value in parts.items():
                totals[name] += value.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        count = len(batches)
        summary = {name: value / count for name, value in totals.items()}
        print(
            f"epoch {epoch + 1}: "
            f"loss={summary['loss']:.4f} "
            f"ground={summary['grounding']:.4f} "
            f"id={summary['identity']:.4f} "
            f"retrieval={summary['retrieval']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "dim": dim,
            },
            output / "last.pt",
        )


if __name__ == "__main__":
    main()
