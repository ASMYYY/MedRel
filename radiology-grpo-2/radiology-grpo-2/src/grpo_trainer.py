from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from .config import GRPOConfig
from .reward_functions import composite_reward


@dataclass
class GRPOTrainer:
    policy: nn.Module
    ref_model: nn.Module
    tokenizer: Any
    config: GRPOConfig

    def __post_init__(self):
        self.policy.to(self.config.device)
        self.ref_model.to(self.config.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    @torch.no_grad()
    def generate_groups(self, prompts: List[str]) -> Dict[str, Any]:
        """
        For each prompt, generate group_size candidates from the current policy.
        """
        device = self.config.device
        group_size = self.config.group_size

        repeated_prompts = []
        prompt_indices = []
        for i, p in enumerate(prompts):
            for _ in range(group_size):
                repeated_prompts.append(p)
                prompt_indices.append(i)
        prompt_indices = torch.tensor(prompt_indices, dtype=torch.long)

        inputs = self.tokenizer(
            repeated_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        gen_outputs = self.policy.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            pad_token_id=pad_id,
        )

        predictions = self.tokenizer.batch_decode(
            gen_outputs, skip_special_tokens=True
        )

        return {
            "gen_input_ids": gen_outputs,        # [B*K, T]
            "prompt_indices": prompt_indices,    # [B*K]
            "predictions": predictions,
        }

    def _sequence_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sequence log-probs and entropy per sample.
        """
        device = self.config.device
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        # Shift to get p(x_t | x_{<t})
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
        prompt_indices: torch.Tensor,
    ) -> torch.Tensor:
        rewards_t = torch.from_numpy(rewards).float().to(self.config.device)
        advantages = torch.zeros_like(rewards_t)

        unique_prompts = prompt_indices.unique()
        for idx in unique_prompts:
            mask = (prompt_indices == idx)
            group_rewards = rewards_t[mask]
            mean = group_rewards.mean()
            std = group_rewards.std(unbiased=False)
            if std < 1e-8:
                std = torch.tensor(1.0, device=mean.device)
            advantages[mask] = (group_rewards - mean) / std
        advantages = torch.clamp(advantages, -5.0, 5.0)
        return advantages

    def compute_grpo_loss(
        self,
        gen_input_ids: torch.Tensor,
        prompt_indices: torch.Tensor,
        references: List[str],
        predictions: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = self.config.device

        # 1. Build references aligned with each generated sample
        sample_refs = [references[i] for i in prompt_indices.tolist()]

        # 2. Rewards
        rewards = composite_reward(predictions, sample_refs)  # np array [B*K]

        # 3. Advantages
        advantages = self._compute_group_advantages(rewards, prompt_indices)

        # 4. Log-probs & entropy
        seq_logp_pi, seq_ent = self._sequence_logprobs(self.policy, gen_input_ids)
        with torch.no_grad():
            seq_logp_ref, _ = self._sequence_logprobs(self.ref_model, gen_input_ids)

        # 5. PPO-style ratio
        log_ratio = seq_logp_pi - seq_logp_ref
        ratio = torch.exp(log_ratio)

        eps = self.config.epsilon_clip
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

        advantages = advantages.to(device)
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.min(obj1, obj2))

        # 6. KL penalty (stabilizing, always >= 0)
        log_ratio = seq_logp_pi - seq_logp_ref
        kl = torch.mean(log_ratio ** 2)
        kl_loss = self.config.kl_coeff * kl


        # 7. Entropy bonus
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
          - 'prompts': List[str]
          - 'references': List[str]
        """
        prompts = batch["prompts"]
        references = batch["references"]

        gen_dict = self.generate_groups(prompts)
        gen_input_ids = gen_dict["gen_input_ids"]
        prompt_indices = gen_dict["prompt_indices"]
        predictions = gen_dict["predictions"]

        loss, stats = self.compute_grpo_loss(
            gen_input_ids=gen_input_ids,
            prompt_indices=prompt_indices,
            references=references,
            predictions=predictions,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return stats
