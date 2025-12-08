import os
import re
import sys
from functools import lru_cache
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Avoid h11 content-length glitches with brotli middleware
os.environ.setdefault("GRADIO_SKIP_BROTLI_COMPRESSION", "1")

import gradio as gr
import torch
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from datetime import datetime
from transformers import AutoImageProcessor, AutoModelForSequenceClassification, AutoTokenizer, VisionEncoderDecoderModel

from models.baseline_supervised import BaselineModel
from preprocess.dataset_loader import default_image_transform, load_image
from utils.tokenizer import load_tokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PT_BASELINE = os.path.join(PROJECT_ROOT, "checkpoints", "baseline", "baseline_best.pt")
PT_GRPO = os.path.join(PROJECT_ROOT, "checkpoints", "grpo", "grpo_step1500.pt")
HF_BASELINE_PRIMARY = os.environ.get(
    "HF_BASELINE_DIR",
    os.path.join(PROJECT_ROOT, "meds_models", "checkpoints", "supervised_vision"),
)
HF_GRPO_PRIMARY = os.environ.get(
    "HF_GRPO_DIR",
    os.path.join(PROJECT_ROOT, "meds_models", "checkpoints", "grpo_vision"),
)
HF_BASELINE_ALT = os.path.join(PROJECT_ROOT, "models", "content", "radiology-grpo", "checkpoints", "supervised_vision")
HF_GRPO_ALT = os.path.join(PROJECT_ROOT, "models", "content", "radiology-grpo", "checkpoints", "grpo_vision")
CHEX_MODEL_ID = os.environ.get("CHEX_MODEL_ID", "IAMJB/chexpert-mimic-cxr-findings-baseline")
REPORTS_CSV = os.path.join(PROJECT_ROOT, "data", "indiana_reports.csv")
PROJ_CSV = os.path.join(PROJECT_ROOT, "data", "indiana_projections.csv")
CHOICES_MODELS = [
    "HF IAMJB/chexpert-mimic-cxr-findings-baseline",
    "HF distilgpt + vit-gpt2",
    "PT ResNet+Transformer",
]


@lru_cache(maxsize=4)
def load_model(tokenizer_name: str, checkpoint_path: str, d_model: int, nhead: int, layers: int) -> Tuple[BaselineModel, object]:
    tokenizer = load_tokenizer(tokenizer_name)
    model = BaselineModel(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id,
        eos_token_id=tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
    ).to(DEVICE)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state["model_state"])
    model.eval()
    return model, tokenizer


def generate_report(image_input, checkpoint_path: str, tokenizer_name: str, max_length: int, d_model: int, nhead: int, layers: int) -> str:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return "Checkpoint not found."
    model, tokenizer = load_model(tokenizer_name, checkpoint_path, d_model, nhead, layers)
    transform = default_image_transform(image_size=224, augment=False)
    if isinstance(image_input, str):
        img = load_image(image_input)
    else:
        img = image_input
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        tokens = model.generate(tensor, max_length=max_length)
    decoded = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
    decoded = sanitize_text(decoded)
    return decoded if decoded else "(no tokens decoded; check checkpoint or image input)"


def sanitize_text(text: str) -> str:
    # keep basic printable ASCII, collapse whitespace
    clean = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    clean = re.sub(r"\s+", " ", clean).strip()
    # collapse pathological single-character repeats (e.g., ℓℓℓ...)
    clean = re.sub(r"(.)\1{10,}", r"\1\1\1", clean)
    return clean


@lru_cache(maxsize=1)
def _load_reference_tables():
    ref_df = None
    proj_df = None
    if os.path.exists(REPORTS_CSV):
        try:
            ref_df = pd.read_csv(REPORTS_CSV)
        except Exception:
            ref_df = None
    if os.path.exists(PROJ_CSV):
        try:
            proj_df = pd.read_csv(PROJ_CSV)
        except Exception:
            proj_df = None
    return ref_df, proj_df


def lookup_reference_report(image_path: str) -> str:
    ref_df, proj_df = _load_reference_tables()
    if ref_df is None or proj_df is None:
        return ""
    filename = os.path.basename(image_path)
    uid_rows = proj_df[proj_df["filename"] == filename]
    if uid_rows.empty:
        return ""
    uid = uid_rows.iloc[0]["uid"]
    rows = ref_df[ref_df["uid"] == uid]
    if rows.empty:
        return ""
    findings = str(rows.iloc[0].get("findings", "") or "")
    impression = str(rows.iloc[0].get("impression", "") or "")
    text = f"Findings: {findings} Impression: {impression}"
    return sanitize_text(text)


@lru_cache(maxsize=2)
def load_hf_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        image_processor = AutoImageProcessor.from_pretrained(path, use_fast=True)
    except OSError:
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(path).to(DEVICE)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    # precompute bad words ([unused*])
    bad_ids = []
    try:
        for i in range(len(tokenizer)):
            tok = tokenizer.convert_ids_to_tokens(i)
            if isinstance(tok, str) and tok.startswith("[unused"):
                bad_ids.append([i])
    except Exception:
        bad_ids = []
    return model, tokenizer, image_processor, bad_ids


def generate_hf(image_input, path: str, max_length: int = 64, num_beams: int = 3, repetition_penalty: float = 1.2) -> str:
    if not path or not os.path.exists(path):
        return "Checkpoint not found."
    model, tokenizer, image_processor, bad_ids = load_hf_model(path)
    gen_cfg = model.generation_config
    if getattr(gen_cfg, "max_length", None):
        max_length = gen_cfg.max_length
    if getattr(gen_cfg, "num_beams", None):
        num_beams = gen_cfg.num_beams
    if getattr(gen_cfg, "repetition_penalty", None):
        repetition_penalty = gen_cfg.repetition_penalty

    if isinstance(image_input, str):
        img = load_image(image_input)
    else:
        img = image_input
    pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_ids if bad_ids else None,
        )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    text = sanitize_text(text)
    return text


def _overlap_f1(ref: str, hyp: str) -> float:
    if not ref or not hyp:
        return 0.0
    r_set, h_set = set(ref.split()), set(hyp.split())
    common = len(r_set & h_set)
    prec = common / max(len(h_set), 1)
    rec = common / max(len(r_set), 1)
    return 2 * prec * rec / max((prec + rec), 1e-8)


def _scores(ref: str, hyp: str) -> Tuple[float, float, float, int]:
    if not ref or not hyp:
        return 0.0, 0.0, 0.0, len(hyp.split())
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth, weights=(0.25, 0.25, 0.25, 0.25))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(ref, hyp)["rougeL"].fmeasure
    overlap = _overlap_f1(ref, hyp)
    return bleu, rouge_l, overlap, len(hyp.split())


def _cider_stub(ref: str, hyp: str) -> float:
    # lightweight placeholder: reuse overlap F1 as a proxy when real CIDEr is unavailable
    return _overlap_f1(ref, hyp)


def _chexbert_stub(ref: str, hyp: str) -> float:
    # placeholder until CheXbert scorer is integrated
    return 0.0


def _radgraph_stub(ref: str, hyp: str) -> float:
    # placeholder until RadGraph scorer is integrated
    return 0.0


@lru_cache(maxsize=1)
def _load_chexbert():
    try:
        tok = AutoTokenizer.from_pretrained(CHEX_MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(CHEX_MODEL_ID).to(DEVICE)
        mdl.eval()
        return tok, mdl
    except Exception:
        return None, None


def chexbert_score(ref: str, hyp: str) -> float:
    tok, mdl = _load_chexbert()
    if tok is None or mdl is None or not ref or not hyp:
        return 0.0
    def _labels(text: str):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**enc).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        labels = set()
        for i, p in enumerate(probs):
            if p.item() >= 0.5:
                labels.add(mdl.config.id2label.get(i, str(i)))
        return labels
    ref_labels = _labels(ref)
    hyp_labels = _labels(hyp)
    if not ref_labels and not hyp_labels:
        return 0.0
    inter = len(ref_labels & hyp_labels)
    return 2 * inter / max((len(ref_labels) + len(hyp_labels)), 1)


def cider_score(ref: str, hyp: str) -> float:
    try:
        scorer = CiderScorer()
        scorer += (hyp.split(), [ref.split()])
        score, _ = scorer.compute_score()
        return float(score)
    except Exception:
        return _cider_stub(ref, hyp)


def radgraph_score(ref: str, hyp: str) -> float:
    # until a full RadGraph parser is wired, approximate with overlap
    return _overlap_f1(ref, hyp)


def run_inference(
    image_input,
    reference_text: str = "",
    baseline_choice: str = CHOICES_MODELS[0],
    grpo_choice: str = CHOICES_MODELS[0],
):
    if image_input is None:
        return ("No image selected or invalid input.", "\n".join([]), "", "", "")

    steps = [
        "Load image and normalize to 224x224.",
        "Encode with ResNet50 backbone to image tokens.",
        "Decode autoregressively with Transformer decoder.",
        "Stop at EOS or max_length.",
    ]

    def resolve(choice: str, hf_primary: str, hf_alt: str, pt_path: str):
        if choice.startswith("HF IAMJB") and os.path.isdir(hf_primary):
            return ("hf", hf_primary)
        if choice.startswith("HF distilgpt") and os.path.isdir(hf_alt):
            return ("hf", hf_alt)
        if choice.startswith("PT ResNet") and pt_path and os.path.exists(pt_path):
            return ("pt", pt_path)
        return (None, None)

    baseline_mode, baseline_ckpt = resolve(baseline_choice, HF_BASELINE_PRIMARY, HF_BASELINE_ALT, PT_BASELINE)
    grpo_mode, grpo_ckpt = resolve(grpo_choice, HF_GRPO_PRIMARY, HF_GRPO_ALT, PT_GRPO)

    status_parts = []
    ref_text = sanitize_text(reference_text) if reference_text else ""
    image_path = image_input if isinstance(image_input, str) else None
    if not ref_text and image_path:
        ref_text = lookup_reference_report(image_path)

    def gen_any(mode: str, ckpt: str, label: str):
        if not mode or not ckpt:
            status_parts.append(f"{label} missing")
            return f"{label} checkpoint not found."
        if mode == "hf":
            return generate_hf(image_input, ckpt, max_length=64, num_beams=2)
        return generate_report(image_input, ckpt, "emilyalsentzer/Bio_ClinicalBERT", max_length=64, d_model=512, nhead=8, layers=6)

    baseline_report = gen_any(baseline_mode, baseline_ckpt, "Baseline")
    grpo_report = gen_any(grpo_mode, grpo_ckpt, "GRPO")

    status = "ok" if not status_parts else "; ".join(status_parts)
    now = datetime.now().strftime("%H:%M:%S")
    status = status + f" | baseline: {baseline_ckpt if baseline_ckpt else 'none'} | grpo: {grpo_ckpt if grpo_ckpt else 'none'} | {now}"

    # collect reports across all available models for metrics
    models_for_metrics = []
    # selected ones
    models_for_metrics.append(("Baseline", baseline_mode, baseline_ckpt, baseline_report))
    models_for_metrics.append(("GRPO", grpo_mode, grpo_ckpt, grpo_report))
    # optional others
    for name, mode, path in [
        ("HF (meds) Baseline", "hf", HF_BASELINE_PRIMARY),
        ("HF (meds) GRPO", "hf", HF_GRPO_PRIMARY),
        ("HF (radiology) Baseline", "hf", HF_BASELINE_ALT),
        ("HF (radiology) GRPO", "hf", HF_GRPO_ALT),
        ("PT Baseline", "pt", PT_BASELINE),
        ("PT GRPO", "pt", PT_GRPO),
    ]:
        if not path or not os.path.exists(path):
            continue
        # skip duplicates already added
        if any(name == m[0] for m in models_for_metrics):
            continue
        txt = gen_any(mode, path, name)
        models_for_metrics.append((name, mode, path, txt))

    metrics_rows: List[List] = []
    if ref_text:
        for name, _, _, txt in models_for_metrics:
            bleu, rouge_l, overlap, tok = _scores(ref_text, txt)
            cider = cider_score(ref_text, txt)
            chex = chexbert_score(ref_text, txt)
            rad = radgraph_score(ref_text, txt)
            metrics_rows.append([
                name,
                round(bleu, 3),
                round(rouge_l, 3),
                round(cider, 3),
                round(chex, 3),
                round(rad, 3),
                round(overlap, 3),
                tok,
            ])

    return (
        status,
        "\n".join(steps),
        ref_text,
        baseline_report,
        grpo_report,
    )


custom_css = """
:root {
  --primary: #500000;
  --secondary: #ffffff;
  --gray-100: #f6f6f6;
  --gray-200: #eaeaea;
  --gray-300: #d1d1d1;
}
body { background: linear-gradient(135deg, #faf7f7 0%, #f6f6f6 50%, #ffffff 100%); color: var(--primary); }
.gradio-container { background: transparent!important; color: var(--primary)!important; }
.gradio-container .main { padding: 10px 14px; }
#inputs-panel, #outputs-panel {
  background: linear-gradient(135deg, #fdf4f4 0%, #f9efef 50%, #fdf4f4 100%);
  color: var(--primary);
  border: 1px solid #f0dcdc;
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.08);
}
#inputs-panel .gr-box, #outputs-panel .gr-box,
#inputs-panel textarea, #outputs-panel textarea,
#inputs-panel input, #outputs-panel input,
#inputs-panel select, #outputs-panel select {
  background: #ffffff!important;
  color: var(--primary)!important;
  border: 1px solid #f0dcdc!important;
  border-radius: 10px!important;
}
#inputs-panel label, #outputs-panel label,
#inputs-panel .block-label, #outputs-panel .block-label,
#inputs-panel .toast-body, #outputs-panel .toast-body {
  color: #2a0000!important;
  font-weight: 650;
  font-size: 1.02em;
}
.btn-primary, button {
  background: var(--primary)!important;
  border: 1px solid var(--gray-300)!important;
  color: var(--secondary)!important;
  border-radius: 12px!important;
}
button:hover {
  background: #732f2f!important;
  color: var(--secondary)!important;
}
.api-docs, a#api-info, a[href*="/api"], a[href*="api/docs"] {
  display: none!important;
}
.gradio-container .prose a#api-info { display: none!important; }
.gradio-container .tabs:last-of-type, .footer, #footer { display: none!important; }
hr { border-color: var(--gray-300); }
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4 { color: #3e3e3e!important; letter-spacing: 0.02em; }
.gradio-container .prose span { color: #3e3e3e!important; }
.gradio-container .label, .gradio-container .block-label, .gradio-container .panel-title {
  color: #3e3e3e!important;
}
.gradio-container label[for*="image"], .gradio-container .image label {
  color: #f5f5f5!important;
}
.gradio-container .image, .gradio-container .image .container, .gradio-container .image input {
  background: #3e3e3e!important;
  border: 1px solid #f0dcdc!important;
}
.gradio-container table {
  background: #fffaf5;
  border: 1px solid #f0dcdc;
}
.gradio-container th {
  background: #f5e3e3;
  color: #2a0000;
  font-weight: 700;
}
.gradio-container textarea, .gradio-container input, .gradio-container select {
  color: #3e3e3e!important;
  background: #ffffff!important;
}
.gradio-container .error, .gradio-container .toast, .gradio-container .feedback {
  color: #b00000!important;
  background: #ffeaea!important;
}
.gradio-container .loading, .gradio-container .loading span {
  color: #b00000!important;
}
.hero {
  background: linear-gradient(135deg, #ffffff 0%, #f6f6f6 50%, #ffffff 100%);
  color: #500000;
  border: 1px solid var(--gray-300);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.08);
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  margin-right: 8px;
  margin-top: 6px;
  background: rgba(80,0,0,0.08);
  color: #500000;
  border: 1px solid rgba(80,0,0,0.18);
  border-radius: 999px;
  font-size: 0.9em;
}
"""

theme = None


with gr.Blocks(title="MedReL: Radiology Report Generation using GRPO with KL-control") as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    gr.HTML(
        """
        <div class='hero' style="background: linear-gradient(135deg, #fdf4f4 0%, #f5e3e3 50%, #fdf4f4 100%); border: 1px solid #f0dcdc; text-align:center;">
          <div style="font-size:2.35em; font-weight:700; color:#3c0000;">MedReL: Radiology Report Generation using GRPO with KL-control</div>
          <div style="margin-top:6px; font-size:1.50em; font-weight:600; color:#3c0000;">GRPO Implementation Demo</div>
          <div style="margin-top:8px; font-size:1.em; color:#500000;">Developers: Asmita Shivling Desai, Sukanya Sahoo</div>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(elem_id="inputs-panel", scale=1):
            gr.Markdown("### Controls")
            with gr.Row():
                run_btn = gr.Button("Generate reports", size="sm")
                clear_btn = gr.ClearButton(
                    components=[],
                    value="Clear all",
                    size="sm",
                )
            with gr.Row():
                image_input = gr.Image(type="filepath", label="Chest X-ray", height=180)
            status = gr.Textbox(label="Status")
            steps_out = gr.Textbox(label="Pipeline steps", lines=4)
        with gr.Column(elem_id="outputs-panel", scale=2):
            gr.Markdown("### Reports")
            reference_out = gr.Textbox(
                label="Reference (from CSV or edited)",
                lines=4,
                placeholder="Auto-filled from CSV if available; otherwise leave blank or paste",
                interactive=True,
            )
            with gr.Row():
                baseline_choice = gr.Dropdown(
                    choices=CHOICES_MODELS,
                    value=CHOICES_MODELS[0],
                    label="Baseline model",
                )
                grpo_choice = gr.Dropdown(
                    choices=CHOICES_MODELS,
                    value=CHOICES_MODELS[0],
                    label="GRPO model",
                )
            with gr.Row():
                hf_base_out = gr.Textbox(label="Baseline", lines=6)
                hf_grpo_out = gr.Textbox(label="GRPO", lines=6)
    run_btn.click(
        fn=run_inference,
        inputs=[image_input, reference_out, baseline_choice, grpo_choice],
        outputs=[
            status,
            steps_out,
            reference_out,
            hf_base_out,
            hf_grpo_out,
        ],
        show_progress="minimal",
    )
    # Clear all fields back to defaults
    clear_btn.add(
        components=[image_input, reference_out, baseline_choice, grpo_choice, status, steps_out, hf_base_out, hf_grpo_out],
        values=[None, "", CHOICES_MODELS[0], CHOICES_MODELS[0], "", "", "", ""],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
