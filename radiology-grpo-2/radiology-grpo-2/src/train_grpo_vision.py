# src/train_grpo_vision.py

from copy import deepcopy
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    ViTImageProcessor,
)

from .config import GRPOConfig
from .dataset import ReportsWithImagesDataset
from .grpo_vision_trainer import VisionGRPOTrainer


def collate_vision_grpo(
    batch: List[Dict],
    image_processor: ViTImageProcessor,
    max_length: int = 128,
):
    """
    Prepare pixel_values and references for GRPO.
    We let VisionEncoderDecoderModel handle text internally.
    """
    images = [b["image"] for b in batch]
    refs = [b["reference"] for b in batch]

    pixel_values = image_processor(
        images=images,
        return_tensors="pt",
    ).pixel_values  # [B, 3, H, W]

    return {
        "pixel_values": pixel_values,
        "references": refs,
    }


def train_grpo_vision(
    data_path: str = "data/iu_reports_with_images.jsonl",
    supervised_ckpt: str = "checkpoints/supervised_vision",
    out_dir: str = "checkpoints/grpo_vision",
    num_steps: int = 200,
):
    gcfg = GRPOConfig()
    device = gcfg.device if torch.cuda.is_available() else "cpu"

    # 1. Load supervised vision model & tokenizer
    policy = VisionEncoderDecoderModel.from_pretrained(supervised_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(supervised_ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.eos_token_id
    policy.config.num_beams = 1


    # 2. Reference model = frozen copy of supervised
    ref_model = deepcopy(policy)

    # 3. Image processor (use base model's processor or original name)
    # If supervised_ckpt doesn't have processor config,
    # it's safe to load from the original captioning model name.
    # Adjust if you used a different encoder-decoder checkpoint.
    from .vision_model import VisionTrainConfig
    vcfg = VisionTrainConfig()
    image_processor = ViTImageProcessor.from_pretrained(
        vcfg.encoder_decoder_model_name
    )

    # 4. Dataset & dataloader
    dataset = ReportsWithImagesDataset(
        jsonl_path=data_path,
        image_transform=None,  # image_processor handles transforms
    )

    def collate_fn(batch):
        return collate_vision_grpo(
            batch,
            image_processor=image_processor,
            max_length=vcfg.max_length,
        )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # start small for T4
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 5. Trainer
    trainer = VisionGRPOTrainer(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        config=gcfg,
    )

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
    print(f"Saved GRPO vision model to {out_dir}")


if __name__ == "__main__":
    train_grpo_vision()
