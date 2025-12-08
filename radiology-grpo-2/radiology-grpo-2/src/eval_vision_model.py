# src/eval_vision_model.py

import json
from pathlib import Path
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from PIL import Image

from .vision_model import VisionTrainConfig


def load_one_sample(jsonl_path: str, idx: int = 0):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    obj = json.loads(lines[idx])
    return obj


def eval_vision_checkpoint(
    model_ckpt: str = "checkpoints/supervised_vision",
    data_path: str = "data/iu_reports_with_images.jsonl",
    sample_idx: int = 0,
):
    cfg = VisionTrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer from your fine-tuned checkpoint
    model = VisionEncoderDecoderModel.from_pretrained(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Load image processor from the base captioning model, not the checkpoint
    image_processor = ViTImageProcessor.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    model.to(device)
    model.eval()

    sample = load_one_sample(data_path, idx=sample_idx)
    img_path = sample["image_path"]
    reference = sample["reference"]
    prompt = sample["prompt"]

    print("Image path:", img_path)
    print("REFERENCE:", reference)

    img = Image.open(img_path).convert("RGB")
    pixel_values = image_processor(
        images=[img],
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values=pixel_values,
            max_length=80,
            num_beams=1,              # force no beam search
            do_sample=False,          # greedy
            no_repeat_ngram_size=4,
            repetition_penalty=1.4,
            length_penalty=1.0,
            early_stopping=True,
        )


    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print("\nGENERATED:")
    print(gen_text)


if __name__ == "__main__":
    # supervised
    eval_vision_checkpoint(
        model_ckpt="checkpoints/supervised_vision",
        data_path="data/iu_reports_with_images.jsonl",
        sample_idx=0,
    )

    # GRPO
    eval_vision_checkpoint(
        model_ckpt="checkpoints/grpo_vision",
        data_path="data/iu_reports_with_images.jsonl",
        sample_idx=0,
    )
