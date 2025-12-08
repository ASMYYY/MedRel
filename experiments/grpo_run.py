import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.optim import AdamW
from tqdm import tqdm

from preprocess.dataset_loader import load_split_csvs
from utils.logger import get_logger
from utils.tokenizer import load_tokenizer
from models.baseline_supervised import BaselineModel
from models.grpo_trainer import GRPOTrainer


def load_baseline(path: str, vocab_size: int, pad: int, bos: int, eos: int, d_model: int, nhead: int, layers: int, device: torch.device):
    model = BaselineModel(
        vocab_size=vocab_size,
        pad_token_id=pad,
        bos_token_id=bos,
        eos_token_id=eos,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
    ).to(device)
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state"])
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("grpo")
    tokenizer = load_tokenizer(args.tokenizer)

    split_csvs = {"train": args.train_csv}
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

    policy = load_baseline(
        path=args.baseline_ckpt,
        vocab_size=len(tokenizer),
        pad=tokenizer.pad_token_id,
        bos=tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id,
        eos=tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        device=device,
    )
    reference_policy = load_baseline(
        path=args.baseline_ckpt,
        vocab_size=len(tokenizer),
        pad=tokenizer.pad_token_id,
        bos=tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id,
        eos=tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        device=device,
    )

    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        tokenizer=tokenizer,
        kl_beta=args.kl_beta,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        max_length=args.max_length,
        min_length=args.min_length,
        group_size=args.group_size,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}")
        pbar = tqdm(loaders["train"], desc=f"grpo {epoch}")
        for batch in pbar:
            metrics = trainer.training_step(batch, optimizer, device)
            global_step += 1
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                rew=f"{metrics['reward_mean']:.3f}",
                kl=f"{metrics['kl']:.3f}",
            )
            if global_step % args.save_steps == 0:
                path = os.path.join(args.output_dir, f"grpo_step{global_step}.pt")
                torch.save({"model_state": policy.state_dict(), "step": global_step}, path)
                logger.info(f"Saved checkpoint to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for MedReL.")
    parser.add_argument("--train_csv", type=str, required=True, help="CSV with train split.")
    parser.add_argument("--image_root", type=str, default="data/mimic-cxr/", help="Root directory for images.")
    parser.add_argument("--baseline_ckpt", type=str, required=True, help="Path to supervised baseline checkpoint.")
    parser.add_argument("--tokenizer", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_beta", type=float, default=0.05)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
