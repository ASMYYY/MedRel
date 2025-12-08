# src/reward_functions.py

from typing import List
import numpy as np
import re
from collections import Counter


def _clean(text: str) -> str:
    # same spirit as train_supervised_vision.clean_report
    text = text.replace("XXXX", "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9.,;:/\-()\s]", " ", text)  # keep basic punct
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> List[str]:
    return _clean(text).split()


def _bigrams(tokens: List[str]) -> List[str]:
    return [" ".join(pair) for pair in zip(tokens, tokens[1:])]


def _unigram_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not ref_tokens or not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = sum(min(pred_counts[w], ref_counts[w]) for w in pred_counts.keys())
    precision = overlap / (len(pred_tokens) + 1e-8)
    recall = overlap / (len(ref_tokens) + 1e-8)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def _bigram_jaccard(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if len(pred_tokens) < 2 or len(ref_tokens) < 2:
        return 0.0
    p_bi = set(_bigrams(pred_tokens))
    r_bi = set(_bigrams(ref_tokens))
    if not p_bi or not r_bi:
        return 0.0
    inter = len(p_bi & r_bi)
    union = len(p_bi | r_bi)
    return float(inter / (union + 1e-8))


def _repetition_penalty(pred_tokens: List[str]) -> float:
    """
    Penalize repeated bigrams like "... thoric.gener changes thoric.gener changes ..."
    """
    if len(pred_tokens) < 2:
        return 0.0
    bigs = _bigrams(pred_tokens)
    counts = Counter(bigs)
    # sum of (count - 1) over bigrams that appear more than once
    extra = sum(c - 1 for c in counts.values() if c > 1)
    # normalize by length
    return extra / (len(bigs) + 1e-8)


def composite_reward(
    predictions: List[str],
    references: List[str],
) -> np.ndarray:
    """
    Reward = 0.7 * unigram F1 + 0.3 * bigram Jaccard
             - 0.3 * repetition_penalty
             - 0.1 * length_penalty (for very long generations)
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        pred_toks = _tokens(pred)
        ref_toks = _tokens(ref)

        f1 = _unigram_f1(pred_toks, ref_toks)
        bigram_score = _bigram_jaccard(pred_toks, ref_toks)

        rep_pen = _repetition_penalty(pred_toks)

        # Length penalty if prediction way longer than reference
        len_ratio = len(pred_toks) / (len(ref_toks) + 1e-8)
        length_pen = max(0.0, len_ratio - 1.5)  # penalize if >150% length

        reward = (
            0.7 * f1
            + 0.3 * bigram_score
            - 0.3 * rep_pen
            - 0.1 * length_pen
        )
        rewards.append(float(reward))

    return np.array(rewards, dtype=np.float32)
