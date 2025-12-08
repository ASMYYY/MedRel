from typing import List

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - optional
    rouge_scorer = None


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    if rouge_scorer is None:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / max(len(scores), 1)
