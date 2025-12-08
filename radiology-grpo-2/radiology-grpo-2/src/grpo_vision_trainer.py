# src/grpo_vision_trainer.py

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor

from .config import GRPOConfig
from .reward_functions import composite_reward


@dataclass
class VisionGRPOTrainer:
    policy: VisionEncoderDecoderModel
    ref_model: VisionEncoderDecoderModel
    tokenizer: AutoTokenizer
    image_processor: ViTImageProcessor
    config: GRPOConfig

    def __post_init__(self):
        self.device = self.config.device if torch.cuda.is_available() else "cpu"

        self.policy.to(self.device)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    @torch.no_grad()
    def generate_groups(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        For each image, generate group_size candidate reports from the current policy.
        pixel_values: [B, 3, H, W]
        Returns flattened pixel_values [B*K, ...] and predictions list of length B*K.
        """
        device = self.device
        group_size = self.config.group_size

        B = pixel_values.size(0)
        # Repeat each image group_size times
        # [B, ...] -> [B*K, ...]
        repeated_pixels = pixel_values.repeat_interleave(group_size, dim=0)
        image_indices = torch.arange(B, device=device).repeat_interleave(
            group_size
        )  # [B*K]

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        gen_ids = self.policy.generate(
            pixel_values=repeated_pixels.to(device),
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=pad_id,
            num_beams=1,              
        )


        predictions = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )

        return {
            "pixel_values": repeated_pixels,  # [B*K, 3, H, W]
            "gen_input_ids": gen_ids,        # [B*K, T]
            "image_indices": image_indices,  # [B*K]
            "predictions": predictions,      # list length B*K
        }

    def _sequence_logprobs(
        self,
        model: VisionEncoderDecoderModel,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sequence log-probs and entropy per sample for encoder-decoder.
        We treat input_ids as the full decoder sequence we want logprob of.
        """
        device = self.device
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Forward through decoder conditioned on images
        outputs = model(
            pixel_values=pixel_values,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        # Shift to get p(x_t | x_{<t}, image)
        labels = input_ids[:, 1:].contiguous()
        log_probs = log_probs[:, :-1, :]  # [B, T-1, V]

        token_logprobs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # [B, T-1]

        label_mask = (labels != self.tokenizer.pad_token_id).float()
        seq_logprobs = (token_logprobs * label_mask).sum(dim=-1)  # [B]

        # Approximate entropy from logits
        probs = log_probs.exp()
        token_entropies = -(probs * log_probs).sum(dim=-1)  # [B, T-1]
        seq_entropy = (token_entropies * label_mask).sum(dim=-1) / (
            label_mask.sum(dim=-1) + 1e-8
        )

        return seq_logprobs, seq_entropy

    def _compute_group_advantages(
        self,
        rewards: np.ndarray,
        image_indices: torch.Tensor,
    ) -> torch.Tensor:
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        advantages = torch.zeros_like(rewards_t)

        unique_images = image_indices.unique()
        for idx in unique_images:
            mask = (image_indices == idx)
            group_rewards = rewards_t[mask]
            mean = group_rewards.mean()
            std = group_rewards.std(unbiased=False)
            if std < 1e-8:
                std = torch.tensor(1.0, device=mean.device)
            advantages[mask] = (group_rewards - mean) / std

        # Clip advantages to avoid huge updates
        advantages = torch.clamp(advantages, -5.0, 5.0)
        return advantages

    def compute_grpo_loss(
        self,
        pixel_values: torch.Tensor,
        gen_input_ids: torch.Tensor,
        image_indices: torch.Tensor,
        references: List[str],
        predictions: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        pixel_values: [B*K, 3, H, W]
        gen_input_ids: [B*K, T]
        image_indices: [B*K]
        references: list length B (one reference per image)
        predictions: list length B*K
        """
        device = self.device

        # Align references per generated sample
        sample_refs = [references[i] for i in image_indices.tolist()]

        # 1. Rewards
        rewards = composite_reward(predictions, sample_refs)  # np array [B*K]

        # 2. Advantages
        advantages = self._compute_group_advantages(rewards, image_indices)

        # 3. Log-probs & entropy (policy and reference)
        seq_logp_pi, seq_ent = self._sequence_logprobs(
            self.policy, pixel_values, gen_input_ids
        )
        with torch.no_grad():
            seq_logp_ref, _ = self._sequence_logprobs(
                self.ref_model, pixel_values, gen_input_ids
            )

        # 4. PPO-style ratio
        log_ratio = seq_logp_pi - seq_logp_ref
        ratio = torch.exp(log_ratio)

        eps = self.config.epsilon_clip
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

        advantages = advantages.to(device)
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.min(obj1, obj2))

        # 5. KL penalty (squared log-ratio, >= 0)
        log_ratio = seq_logp_pi - seq_logp_ref
        kl = torch.mean(log_ratio ** 2)
        kl_loss = self.config.kl_coeff * kl


        # 6. Entropy bonus
        entropy_loss = -self.config.entropy_coeff * seq_ent.mean()

        total_loss = policy_loss + kl_loss + entropy_loss

        stats = {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl": float(kl.item()),
            "entropy": float(seq_ent.mean().item()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
        }
        return total_loss, stats

    def training_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        batch:
          - 'pixel_values': [B, 3, H, W]
          - 'references':  List[str] (one per image)
        """
        pixel_values = batch["pixel_values"].to(self.device)
        references: List[str] = batch["references"]

        # 1. Generate candidates per image
        gen_dict = self.generate_groups(pixel_values)
        rep_pixels = gen_dict["pixel_values"]
        gen_input_ids = gen_dict["gen_input_ids"]
        image_indices = gen_dict["image_indices"]
        predictions = gen_dict["predictions"]

        # 2. Compute GRPO loss
        loss, stats = self.compute_grpo_loss(
            pixel_values=rep_pixels.to(self.device),
            gen_input_ids=gen_input_ids,
            image_indices=image_indices,
            references=references,
            predictions=predictions,
        )

        # 3. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return stats
