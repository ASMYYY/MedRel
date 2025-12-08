from typing import Optional

from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT", use_fast: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    if getattr(tokenizer, "cls_token", None) is None and tokenizer.sep_token is not None:
        tokenizer.cls_token = tokenizer.sep_token
    return tokenizer


class SimpleTokenizer:
    """
    Minimal whitespace tokenizer for debugging when transformers is unavailable.
    Token ids are not stable; use only for smoke tests.
    """

    def __init__(self, vocab: Optional[dict] = None, pad_id: int = 0, unk_id: int = 1):
        self.vocab = vocab or {}
        self.pad_token_id = pad_id
        self.unk_token_id = unk_id

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 256,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
    ):
        tokens = text.strip().split()
        if truncation:
            tokens = tokens[:max_length]
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = ids[: max_length - 2]
            ids = [101] + ids + [102]
        attention = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": attention}

    def pad(self, encoded_inputs, padding=True, return_tensors=None):
        input_ids = encoded_inputs["input_ids"]
        attention = encoded_inputs["attention_mask"]
        max_len = max(len(seq) for seq in input_ids) if padding else 0
        padded_ids, padded_attn = [], []
        for ids, attn in zip(input_ids, attention):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attn.append(attn + [0] * pad_len)
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor(padded_ids, dtype=torch.long), "attention_mask": torch.tensor(padded_attn, dtype=torch.long)}
        return {"input_ids": padded_ids, "attention_mask": padded_attn}
