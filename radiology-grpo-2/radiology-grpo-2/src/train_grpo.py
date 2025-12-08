import os
from pathlib import Path
from copy import deepcopy
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import TrainConfig, GRPOConfig
from .dataset import ReportsDataset
from .grpo_trainer import GRPOTrainer


def collate_prompts_refs(batch: List[Dict]):
    prompts = [b["prompt"] for b in batch]
    refs = [b["reference"] for b in batch]
    return {"prompts": prompts, "references": refs}


def train_grpo(
    data_path: str = "data/toy_reports.jsonl",
    supervised_ckpt: str = "checkpoints/supervised",
    out_dir: str = "checkpoints/grpo",
    num_steps: int = 50,
):
    tcfg = TrainConfig()
    gcfg = GRPOConfig()
    device = gcfg.device if torch.cuda.is_available() else "cpu"

    # Load tokenizer + supervised model
    tokenizer = AutoTokenizer.from_pretrained(supervised_ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy = AutoModelForCausalLM.from_pretrained(supervised_ckpt)
    policy.config.pad_token_id = tokenizer.pad_token_id

    # Reference model (frozen)
    ref_model = deepcopy(policy)

    dataset = ReportsDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=collate_prompts_refs,
    )

    trainer = GRPOTrainer(policy=policy, ref_model=ref_model,
                          tokenizer=tokenizer, config=gcfg)

    step = 0
    while step < num_steps:
        for batch in dataloader:
            stats = trainer.training_step(batch)
            step += 1
            print(
                f"[Step {step}] "
                f"loss={stats['loss']:.4f} "
                f"reward_mean={stats['reward_mean']:.4f} "
                f"kl={stats['kl']:.4f} "
                f"entropy={stats['entropy']:.4f}"
            )
            if step >= num_steps:
                break

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved GRPO-tuned model to {out_dir}")


if __name__ == "__main__":
    train_grpo()
