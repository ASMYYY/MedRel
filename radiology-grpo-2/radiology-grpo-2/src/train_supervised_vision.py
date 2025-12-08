# src/train_supervised_vision.py

import os
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import ReportsWithImagesDataset
from .vision_model import load_vision_model_and_processor, VisionTrainConfig


def clean_report(text: str) -> str:
    text = text.replace("XXXX", "")          # remove redactions
    text = re.sub(r"\s+", " ", text)         # collapse spaces
    return text.strip()


def collate_vision(batch, tokenizer, image_processor, max_length: int):
    images = [b["image"] for b in batch]
    prompts = [b["prompt"] for b in batch]  # e.g. "FINDINGS: "
    refs = [b["reference"] for b in batch]

    # Encode images -> pixel_values
    pixel_values = image_processor(
        images=images,
        return_tensors="pt",
    ).pixel_values  # [B, 3, H, W]

    # Decoder text: "FINDINGS: <report>"
    texts = [clean_report(p + " " + r) for p, r in zip(prompts, refs)]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def train_supervised_vision(
    data_path: str = "data/iu_reports_with_images.jsonl",
    out_dir: str = "checkpoints/supervised_vision",
):
    cfg = VisionTrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Load model, tokenizer, and image processor
    model, tokenizer, image_processor = load_vision_model_and_processor(cfg)
    model.to(device)

    # Freeze the vision encoder (ViT) – train mainly decoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Dataset with image paths
    full_dataset = ReportsWithImagesDataset(
        jsonl_path=data_path,
        image_transform=None,
    )

    # 90/10 train/val split
    val_frac = 0.1
    n_total = len(full_dataset)
    n_val = max(1, int(val_frac * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    def collate_fn(batch):
        coll = collate_vision(
            batch, tokenizer, image_processor, max_length=cfg.max_length
        )
        coll["pixel_values"] = coll["pixel_values"]
        return coll

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        print(f"\n=== Vision Supervised Epoch {epoch+1}/{cfg.num_epochs} ===")
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = batch["pixel_values"].size(0)
            total_loss += loss.item() * bs
            total_count += bs

            # Print every 100 steps (or last step)
            if step % 100 == 0 or step == len(train_loader):
                avg_loss = total_loss / total_count
                print(f"  step {step}/{len(train_loader)}  train_loss={avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                bs = batch["pixel_values"].size(0)
                val_loss += loss.item() * bs
                val_count += bs
        val_loss = val_loss / max(1, val_count)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved supervised vision model to {out_dir}")


if __name__ == "__main__":
    train_supervised_vision()
