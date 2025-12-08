from typing import List

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
except ImportError:  # pragma: no cover - optional
    corpus_bleu = None
    SmoothingFunction = None


def compute_bleu(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """
    references: list of list of reference token lists (multiple refs per hyp)
    hypotheses: list of hypothesis token lists
    """
    if corpus_bleu is None:
        return 0.0
    smooth = SmoothingFunction().method1 if SmoothingFunction else None
    return corpus_bleu(references, hypotheses, smoothing_function=smooth)
