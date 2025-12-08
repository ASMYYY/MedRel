from typing import Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import TrainConfig


def load_tokenizer_and_model(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Some GPT-like models don't have a pad token; reuse eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


class SupervisedCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Concatenate prompt + reference as one sequence for language modeling
        text_list = []
        for sample in batch:
            prompt = sample["prompt"]
            ref = sample["reference"]
            full_text = prompt + " " + ref
            text_list.append(full_text)

        encodings = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Standard LM objective: labels = input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
