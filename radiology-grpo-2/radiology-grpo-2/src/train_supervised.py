import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import ReportsDataset
from .models import load_tokenizer_and_model, SupervisedCollator


def train_supervised(
    data_path: str = "data/toy_reports.jsonl",
    out_dir: str = "checkpoints/supervised",
):
    cfg = TrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_tokenizer_and_model(cfg)
    model.to(device)

    dataset = ReportsDataset(data_path)
    collator = SupervisedCollator(tokenizer, max_length=cfg.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    model.train()
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dataloader, desc=f"Supervised Epoch {epoch+1}")
        total_loss = 0.0
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch["input_ids"].size(0)
            avg_loss = total_loss / ((pbar.n + 1) * cfg.batch_size)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved supervised model to {out_dir}")


if __name__ == "__main__":
    train_supervised()
