from typing import Dict, List

from .bleu import compute_bleu
from .cider import compute_cider
from .chexbert import compute_chexbert_f1
from .radgraph import compute_radgraph_f1
from .rouge import compute_rouge_l


def simple_overlap_score(references_text: List[str], hypotheses_text: List[str]) -> float:
    """
    Basic unigram overlap to avoid all-zero rewards on small datasets or placeholder metrics.
    """
    scores = []
    for ref, hyp in zip(references_text, hypotheses_text):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        ref_set, hyp_set = set(ref_tokens), set(hyp_tokens)
        common = len(ref_set & hyp_set)
        precision = common / max(len(hyp_set), 1)
        recall = common / max(len(ref_set), 1)
        f1 = 2 * precision * recall / max((precision + recall), 1e-8)
        scores.append(f1)
    return sum(scores) / max(len(scores), 1)


def contradiction_penalty(references: List[str], hypotheses: List[str]) -> float:
    penalty = 0.0
    neg_tokens = {"no", "without", "absent", "negative"}
    for ref, hyp in zip(references, hypotheses):
        ref_neg = any(tok in ref.lower() for tok in neg_tokens)
        hyp_neg = any(tok in hyp.lower() for tok in neg_tokens)
        if ref_neg != hyp_neg:
            penalty += 1.0
    if not references:
        return 0.0
    return penalty / len(references)


def compute_composite_reward(
    references_tokenized: List[List[List[str]]],
    hypotheses_tokenized: List[List[str]],
    references_text: List[str],
    hypotheses_text: List[str],
    weights: Dict[str, float] = None,
) -> float:
    if weights is None:
        weights = {
            "bleu": 0.2,
            "rouge": 0.2,
            "cider": 0.2,
            "chexbert": 0.1,
            "radgraph": 0.1,
            "overlap": 0.2,
            "penalty": 0.0,
        }
    bleu = compute_bleu(references_tokenized, hypotheses_tokenized)
    rouge = compute_rouge_l(references_text, hypotheses_text)
    cider = compute_cider(references_tokenized, [" ".join(h) for h in hypotheses_tokenized])
    chex = compute_chexbert_f1(references_text, hypotheses_text)
    rad = compute_radgraph_f1(references_text, hypotheses_text)
    overlap = simple_overlap_score(references_text, hypotheses_text)
    pen = contradiction_penalty(references_text, hypotheses_text)
    reward = (
        weights.get("bleu", 0) * bleu
        + weights.get("rouge", 0) * rouge
        + weights.get("cider", 0) * cider
        + weights.get("chexbert", 0) * chex
        + weights.get("radgraph", 0) * rad
        + weights.get("overlap", 0) * overlap
        - weights.get("penalty", 0) * pen
    )
    return reward
