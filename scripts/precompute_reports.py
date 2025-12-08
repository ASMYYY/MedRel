import argparse
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from models.baseline_supervised import BaselineModel
from preprocess.dataset_loader import default_image_transform, load_image
from utils.tokenizer import load_tokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_pt_model(ckpt_path: str, tokenizer_name: str, d_model: int = 512, nhead: int = 8, layers: int = 6) -> Tuple[BaselineModel, object]:
    tok = load_tokenizer(tokenizer_name)
    model = BaselineModel(
        vocab_size=len(tok),
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id,
        eos_token_id=tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, tok


def load_hf_model(path: str):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    processor = AutoImageProcessor.from_pretrained(path, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(path).to(DEVICE)
    if model.config.pad_token_id is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok, processor


def gen_pt(model: BaselineModel, tok, img, max_length: int = 64) -> str:
    transform = default_image_transform(image_size=224, augment=False)
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        ids = model.generate(tensor, max_length=max_length)[0].tolist()
    return sanitize(tok.decode(ids, skip_special_tokens=True))


def gen_hf(model, tok, processor, img, max_length: int = 64, num_beams: int = 3) -> str:
    pixels = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        out = model.generate(pixel_values=pixels, max_length=max_length, num_beams=num_beams)
    return sanitize(tok.decode(out[0], skip_special_tokens=True))


def metrics(ref: str, hyp: str) -> Dict[str, float]:
    if not ref or not hyp:
        return dict(bleu4=0.0, rougeL=0.0, cider=0.0, overlap=0.0)
    smooth = SmoothingFunction().method1
    bleu4 = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth, weights=(0.25, 0.25, 0.25, 0.25))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = scorer.score(ref, hyp)["rougeL"].fmeasure
    try:
        cs = CiderScorer()
        cs += (hyp.split(), [ref.split()])
        cider, _ = cs.compute_score()
    except Exception:
        cider = 0.0
    rset, hset = set(ref.split()), set(hyp.split())
    common = len(rset & hset)
    prec = common / max(len(hset), 1)
    rec = common / max(len(rset), 1)
    overlap = 2 * prec * rec / max((prec + rec), 1e-8)
    return dict(bleu4=bleu4, rougeL=rougeL, cider=cider, overlap=overlap)


def main():
    ap = argparse.ArgumentParser(description="Precompute reports/metrics for multiple models.")
    ap.add_argument("--csv", required=True, help="CSV with image_path and report_text/reference.")
    ap.add_argument("--image_root", default="data", help="Root for image paths.")
    ap.add_argument("--max_samples", type=int, default=3000, help="Max samples to process.")
    ap.add_argument("--out_csv", default="data/precomputed_reports.csv", help="Output CSV path.")
    args = ap.parse_args()

    rows = pd.read_csv(args.csv)
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows.sample(args.max_samples, random_state=42).reset_index(drop=True)

    models = [
        ("hf_meds_baseline", "hf", os.path.join(PROJECT_ROOT, "meds_models", "checkpoints", "supervised_vision")),
        ("hf_meds_grpo", "hf", os.path.join(PROJECT_ROOT, "meds_models", "checkpoints", "grpo_vision")),
        ("hf_vilmedic_baseline", "hf", os.path.join(PROJECT_ROOT, "models", "content", "radiology-grpo", "checkpoints", "supervised_vision")),
        ("hf_vilmedic_grpo", "hf", os.path.join(PROJECT_ROOT, "models", "content", "radiology-grpo", "checkpoints", "grpo_vision")),
        ("pt_baseline", "pt", os.path.join(PROJECT_ROOT, "checkpoints", "baseline", "baseline_best.pt")),
        ("pt_grpo", "pt", os.path.join(PROJECT_ROOT, "checkpoints", "grpo", "grpo_step1500.pt")),
    ]

    # preload models
    hf_cache, pt_cache = {}, {}
    for name, mtype, path in models:
        if mtype == "hf" and os.path.isdir(path):
            hf_cache[name] = load_hf_model(path)
        elif mtype == "pt" and os.path.exists(path):
            pt_cache[name] = load_pt_model(path, "emilyalsentzer/Bio_ClinicalBERT")

    out_records = []
    for _, row in rows.iterrows():
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(args.image_root, img_path)
        try:
            img = load_image(img_path)
        except Exception:
            continue
        ref = sanitize(row.get("report_text", row.get("raw_report", "")) or "")
        rec = {"image_path": img_path, "reference": ref}
        for name, mtype, path in models:
            if mtype == "hf":
                tpl = hf_cache.get(name)
                if tpl:
                    model, tok, proc = tpl
                    hyp = gen_hf(model, tok, proc, img)
                else:
                    hyp = ""
            else:
                tpl = pt_cache.get(name)
                if tpl:
                    model, tok = tpl
                    hyp = gen_pt(model, tok, img)
                else:
                    hyp = ""
            rec[f"{name}_text"] = hyp
            sc = metrics(ref, hyp)
            rec[f"{name}_bleu4"] = sc["bleu4"]
            rec[f"{name}_rougeL"] = sc["rougeL"]
            rec[f"{name}_cider"] = sc["cider"]
            rec[f"{name}_overlap"] = sc["overlap"]
        out_records.append(rec)

    pd.DataFrame(out_records).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_records)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
