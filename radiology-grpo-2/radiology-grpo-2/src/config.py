from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Text-only LM (for supervised + GRPO text)
    model_name: str = "microsoft/biogpt"
    max_length: int = 128
    batch_size: int = 4      # you can bump to 8 on big GPU
    num_epochs: int = 3      # a bit more training now that you're on HPRC
    lr: float = 3e-5         # slightly smaller for BioGPT
    weight_decay: float = 0.01
    device: str = "cuda"


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 40
    temperature: float = 0.7
    top_p: float = 0.9
    lr: float = 5e-6
    weight_decay: float = 0.01
    epsilon_clip: float = 0.1
    kl_coeff: float = 0.1
    entropy_coeff: float = 0.001
    max_grad_norm: float = 1.0
    device: str = "cuda"
