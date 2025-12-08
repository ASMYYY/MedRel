# src/vision_model.py

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    ViTImageProcessor,
)
from dataclasses import dataclass

from .config import TrainConfig


@dataclass
class VisionTrainConfig(TrainConfig):
    image_size: int = 224
    # use CheXpert+MIMIC CXR findings baseline
    encoder_decoder_model_name: str = "IAMJB/chexpert-mimic-cxr-findings-baseline"
    num_epochs: int = 8          # you have more compute now
    batch_size: int = 4          # try 4–8, reduce if OOM
    max_length: int = 80         # slightly longer findings
    lr: float = 2e-5             # smaller LR for a strong pretrained model


def load_vision_model_and_processor(cfg: VisionTrainConfig):
    """
    Load a VisionEncoderDecoder model for chest X-ray -> findings generation,
    using IAMJB/chexpert-mimic-cxr-findings-baseline as backbone.
    """

    # VisionEncoderDecoder (ViT encoder + BERT-ish decoder)
    model = VisionEncoderDecoderModel.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    # Image processor for ViT encoder
    image_processor = ViTImageProcessor.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    # Tokenizer for decoder (BERT-style vocab)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    # ---- Special token ids ----
    # For BERT-like tokenizers, we typically have:
    #  - cls_token_id
    #  - sep_token_id
    #  - pad_token_id
    #
    # Decoder config needs:
    #  - pad_token_id
    #  - decoder_start_token_id
    #  - eos_token_id

    # pad_token_id
    if tokenizer.pad_token_id is None:
        # fall back if pad not set
        if tokenizer.sep_token_id is not None:
            tokenizer.pad_token_id = tokenizer.sep_token_id
        elif tokenizer.cls_token_id is not None:
            tokenizer.pad_token_id = tokenizer.cls_token_id

    pad_id = tokenizer.pad_token_id
    model.config.pad_token_id = pad_id

    # decoder_start_token_id (usually CLS for encoder-decoder with BERT decoder)
    if tokenizer.cls_token_id is not None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    else:
        # fallback to pad/eos if no CLS
        model.config.decoder_start_token_id = pad_id

    # eos_token_id (usually SEP)
    if tokenizer.sep_token_id is not None:
        model.config.eos_token_id = tokenizer.sep_token_id
    else:
        model.config.eos_token_id = pad_id

    # Generation defaults
    model.config.max_length = cfg.max_length
    model.config.num_beams = 1    # keep 1 to avoid _reorder_cache issues with GPT2-like decoders

    return model, tokenizer, image_processor
