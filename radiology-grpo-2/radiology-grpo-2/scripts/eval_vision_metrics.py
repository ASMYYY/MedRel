# scripts/eval_vision_metrics.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor

from src.dataset import ReportsWithImagesDataset
from src.vision_model import VisionTrainConfig

# We'll use `evaluate` for BLEU and ROUGE
# pip install evaluate
import evaluate


def collate_for_eval(batch, image_processor):
    images = [b["image"] for b in batch]
    refs = [b["reference"] for b in batch]

    pixel_values = image_processor(
        images=images,
        return_tensors="pt",
    ).pixel_values

    return {
        "pixel_values": pixel_values,
        "references": refs,
    }


def run_eval(
    model_ckpt: str,
    data_path: str = "data/iu_reports_with_images.jsonl",
    num_samples: int = 200,
):
    cfg = VisionTrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_ckpt}")
    model = VisionEncoderDecoderModel.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # image processor from base captioning model
    from transformers import ViTImageProcessor
    image_processor = ViTImageProcessor.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    dataset = ReportsWithImagesDataset(
        jsonl_path=data_path,
        image_transform=None,
    )

    # subset for speed
    if num_samples is not None and num_samples < len(dataset):
        subset_indices = list(range(num_samples))
        from torch.utils.data import Subset
        dataset = Subset(dataset, subset_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_for_eval(b, image_processor),
    )

    model.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {model_ckpt}"):
            pixel_values = batch["pixel_values"].to(device)
            refs = batch["references"]

            gen_ids = model.generate(
                pixel_values=pixel_values,
                max_length=cfg.max_length,
                num_beams=1,              #disable beam search
                do_sample=False,          # greedy
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )


            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    print(f"Collected {len(all_preds)} predictions.")

    # ---- BLEU and ROUGE via `evaluate` ----
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    # BLEU expects list[str] predictions, list[list[str]] references
    bleu_result = bleu.compute(
        predictions=all_preds,
        references=[[r] for r in all_refs],
    )

    rouge_result = rouge.compute(
        predictions=all_preds,
        references=all_refs,
    )

    print("\n=== METRICS for", model_ckpt, "===")
    print("BLEU:", bleu_result["bleu"])
    print("ROUGE-L:", rouge_result["rougeL"])
    print("Full BLEU:", bleu_result)
    print("Full ROUGE:", rouge_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g. checkpoints/supervised_vision)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/iu_reports_with_images.jsonl",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="How many samples from IU dataset to evaluate",
    )

    args = parser.parse_args()
    run_eval(
        model_ckpt=args.model_ckpt,
        data_path=args.data_path,
        num_samples=args.num_samples,
    )
