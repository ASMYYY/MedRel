import argparse
import os
import sys
from typing import Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess.dataset_loader import load_split_csvs
from utils.logger import get_logger
from utils.tokenizer import load_tokenizer
from models.baseline_supervised import BaselineModel


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device, pad_token_id: int):
    images = batch["images"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    # shift for decoder: teacher forcing targets exclude first token
    decoder_inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:].clone()
    labels[labels == pad_token_id] = pad_token_id
    attn_mask = attention_mask[:, :-1]
    return images, decoder_inputs, labels, attn_mask


def evaluate(model: BaselineModel, loader: DataLoader, device: torch.device, pad_token_id: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images, decoder_inputs, labels, attn_mask = prepare_batch(batch, device, pad_token_id)
            outputs = model(images=images, input_ids=decoder_inputs, attention_mask=attn_mask)
            logits = outputs["logits"]
            loss = model.compute_loss(logits, labels)
            token_count = labels.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count
    model.train()
    return total_loss / max(total_tokens, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("baseline")
    tokenizer = load_tokenizer(args.tokenizer)

    split_csvs = {"train": args.train_csv, "val": args.val_csv}
    loaders = load_split_csvs(
        split_csvs=split_csvs,
        image_root=args.image_root,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        image_size=args.image_size,
        augment_train=True,
    )

    model = BaselineModel(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id,
        eos_token_id=tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}")
        model.train()
        pbar = tqdm(loaders["train"], desc=f"train {epoch}")
        for batch in pbar:
            images, decoder_inputs, labels, attn_mask = prepare_batch(batch, device, tokenizer.pad_token_id)
            outputs = model(images=images, input_ids=decoder_inputs, attention_mask=attn_mask)
            logits = outputs["logits"]
            loss = model.compute_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        val_loss = evaluate(model, loaders["val"], device, tokenizer.pad_token_id)
        logger.info(f"Val loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(args.output_dir, "baseline_best.pt")
            torch.save({"model_state": model.state_dict(), "tokenizer": args.tokenizer}, save_path)
            logger.info(f"Saved best model to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised baseline training for MedReL (ViLMedic-style).")
    parser.add_argument("--train_csv", type=str, required=True, help="CSV with train split and image/report paths.")
    parser.add_argument("--val_csv", type=str, required=True, help="CSV with val split.")
    parser.add_argument("--image_root", type=str, default="data/mimic-cxr/", help="Root directory for images.")
    parser.add_argument("--tokenizer", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Tokenizer name or path.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
