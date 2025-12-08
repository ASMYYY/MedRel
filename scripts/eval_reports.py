import argparse
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from preprocess.dataset_loader import default_image_transform, load_image
from models.baseline_supervised import BaselineModel
from utils.tokenizer import load_tokenizer


def sanitize_text(text: str) -> str:
    return " ".join(text.strip().split())


def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]
    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(refs_tok, hyps_tok, smoothing_function=smooth) if hyps_tok else 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    rouge_l = sum(rouge_scores) / max(len(rouge_scores), 1)
    overlap_scores = []
    for r, h in zip(references, hypotheses):
        r_set, h_set = set(r.split()), set(h.split())
        common = len(r_set & h_set)
        prec = common / max(len(h_set), 1)
        rec = common / max(len(r_set), 1)
        f1 = 2 * prec * rec / max((prec + rec), 1e-8)
        overlap_scores.append(f1)
    overlap = sum(overlap_scores) / max(len(overlap_scores), 1)
    return {"bleu": bleu, "rougeL": rouge_l, "overlap": overlap}


def load_pt_model(ckpt_path: str, tokenizer_name: str, d_model: int = 512, nhead: int = 8, layers: int = 6, device: str = "cpu") -> Tuple[BaselineModel, object]:
    tok = load_tokenizer(tokenizer_name)
    model = BaselineModel(
        vocab_size=len(tok),
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id,
        eos_token_id=tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, tok


def gen_pt(model: BaselineModel, tokenizer, image, device: str, max_length: int = 64) -> str:
    transform = default_image_transform(image_size=224, augment=False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.generate(tensor, max_length=max_length)[0].tolist()
    return sanitize_text(tokenizer.decode(tokens, skip_special_tokens=True))


def load_hf_model(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    try:
        processor = AutoImageProcessor.from_pretrained(path)
    except OSError:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = VisionEncoderDecoderModel.from_pretrained(path).to(device)
    model.eval()
    return model, tok, processor


def gen_hf(model, tok, processor, image, device: str, max_length: int = 64, num_beams: int = 3) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out_ids = model.generate(pixel_values=pixel_values, max_length=max_length, num_beams=num_beams)
    return sanitize_text(tok.decode(out_ids[0], skip_special_tokens=True))


def evaluate(csv_path: str, image_root: str, num_samples: int, ckpts: Dict[str, Dict[str, str]], device: str):
    df = pd.read_csv(csv_path)
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)

    rows = []
    for _, row in df.iterrows():
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(image_root, img_path)
        try:
            img = load_image(img_path)
        except Exception:
            continue
        ref = sanitize_text(str(row.get("report_text", row.get("raw_report", "")) or ""))
        rows.append((img, ref))

    results = {}
    for name, info in ckpts.items():
        mode = info["mode"]
        path = info["path"]
        if mode == "pt":
            model, tok = load_pt_model(path, info.get("tokenizer", "emilyalsentzer/Bio_ClinicalBERT"), device=device)
            hyps = [gen_pt(model, tok, img, device) for img, _ in rows]
        else:
            model, tok, proc = load_hf_model(path, device)
            hyps = [gen_hf(model, tok, proc, img, device) for img, _ in rows]
        refs = [r for _, r in rows]
        results[name] = compute_metrics(refs, hyps)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline/GRPO checkpoints (HF or PT) on a CSV split.")
    parser.add_argument("--csv", type=str, required=True, help="CSV with image_path and report_text/raw_report.")
    parser.add_argument("--image_root", type=str, default="data", help="Root for image paths.")
    parser.add_argument("--num_samples", type=int, default=200, help="Limit samples (0=all).")
    parser.add_argument("--pt_baseline", type=str, help="Path to PT baseline .pt checkpoint.")
    parser.add_argument("--pt_grpo", type=str, help="Path to PT GRPO .pt checkpoint.")
    parser.add_argument("--hf_baseline", type=str, help="Path to HF VisionEncoderDecoder baseline dir.")
    parser.add_argument("--hf_grpo", type=str, help="Path to HF VisionEncoderDecoder GRPO dir.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = {}
    if args.pt_baseline:
        ckpts["pt_baseline"] = {"mode": "pt", "path": args.pt_baseline}
    if args.pt_grpo:
        ckpts["pt_grpo"] = {"mode": "pt", "path": args.pt_grpo}
    if args.hf_baseline:
        ckpts["hf_baseline"] = {"mode": "hf", "path": args.hf_baseline}
    if args.hf_grpo:
        ckpts["hf_grpo"] = {"mode": "hf", "path": args.hf_grpo}
    if not ckpts:
        raise SystemExit("No checkpoints provided.")

    results = evaluate(args.csv, args.image_root, args.num_samples, ckpts, device)
    for name, metrics in results.items():
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
