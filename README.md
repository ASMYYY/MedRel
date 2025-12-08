# MedReL: Radiology Report Generation with Supervised Baseline and GRPO

## Overview

### **Supervised Vision–Language Baseline**
MedReL begins with a supervised encoder–decoder model based on the pretrained  
**`IAMJB/chexpert-mimic-cxr-findings-baseline`** checkpoint:

- **ViT-based image encoder**
- **BERT-style text decoder**
- Encoder is **frozen**; training updates only the decoder
- Dataset: **Indiana University Chest X-ray dataset**, Findings-only generation
- Training uses cross-entropy with standard vision–language preprocessing

This supervised model serves as the reference policy for reinforcement learning.


### **GRPO Fine-Tuning with KL-Control**
We refine the supervised model using **Group Relative Policy Optimization (GRPO)**:

- Generates multiple candidate reports per image (group sampling)
- Computes **group-normalized advantages**
- Uses a **PPO-style clipped objective**
- Applies **KL-control** to constrain divergence from the supervised baseline

#### **Reward Function (Implemented)**
The GRPO reward is a lightweight lexical reward composed of:

- **Unigram F1 overlap** between prediction and reference  
- **Bigram Jaccard similarity**
- **Repetition penalty** to discourage bigram loops
- **Length penalty** for excessively long outputs


### **Model Evaluation**
We include scripts to evaluate supervised and GRPO-tuned checkpoints using:

- **BLEU**
- **ROUGE-L**

Generation and evaluation operate on:
- checkpoints/supervised_vision
- checkpoints/grpo_vision


Project layout
--------------
- `data/`
  - `images/images_normalized/`: IU X-ray images for the demo.
  - `indiana/iu_train.csv`, `indiana/iu_val.csv`: 80/20 split; `image_path` is relative to `data/`, `report_text` combines Findings/Impression.
  - `mimic-cxr/`: placeholder if you switch to MIMIC-CXR.
- `preprocess/`: dataset loader, image transforms, section extraction, tokenization.
- `models/`: vision encoder, text decoder, baseline wrapper, GRPO trainer.
- `rewards/`: metric stubs; `composite_reward.py` includes an overlap reward and zero penalty by default.
- `utils/`: tokenizer loader (Bio_ClinicalBERT), logger, plotter.
- `experiments/`: `baseline_run.py` (supervised), `grpo_run.py` (GRPO).
- `ui/`: `app.py` Gradio UI (baseline vs GRPO) with CSV reference lookup and optional override.
- `models/content/radiology-grpo/checkpoints/{supervised_vision,grpo_vision}`: HF VisionEncoderDecoder checkpoints used by the UI.

Setup
-----
1) Use Python 3.11. Create venv: `py -3.11 -m venv .venv && .\.venv\Scripts\Activate.ps1`
2) Install torch (pick one):
   - GPU: `python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`
   - CPU: `python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu`
3) Install remaining deps: `python -m pip install -r requirements.txt --no-deps`
4) Ensure CSV paths point to images (IU: `image_path` like `images/images_normalized/xxx.png`, `--image_root data`).

Training
--------
- Baseline:
  ```
  python experiments/baseline_run.py --train_csv data/indiana/iu_train.csv --val_csv data/indiana/iu_val.csv --image_root data --output_dir checkpoints/baseline --num_workers 0
  ```
- GRPO (after baseline):
  ```
  python experiments/grpo_run.py --train_csv data/indiana/iu_train.csv --image_root data --baseline_ckpt checkpoints/baseline/baseline_best.pt --output_dir checkpoints/grpo --num_workers 0 --group_size 4 --top_p 0.8 --temperature 0.7 --min_length 5
  ```
Expected checkpoints:
- Baseline: `checkpoints/baseline/baseline_best.pt`
- GRPO: `checkpoints/grpo/grpo_stepXXXX.pt` (latest used in UI)

UI
--
```
python ui/app.py
```
Two ways to use the UI:
- Default (HF checkpoints): UI auto-loads `models/content/radiology-grpo/checkpoints/{supervised_vision,grpo_vision}`. Upload an image; reference text is pulled from `indiana_reports.csv`/`indiana_projections.csv` by filename match, and GRPO falls back to baseline if missing.
- If you prefer your own trained .pt checkpoints (baseline/grpo in `checkpoints/`), adjust the UI loader to point to those paths instead of the HF folders.

Metrics (HF or PT checkpoints)
------------------------------
Use the evaluator:
```
python scripts/eval_reports.py --csv data/indiana/iu_val.csv --image_root data \
  --hf_baseline models/content/radiology-grpo/checkpoints/supervised_vision \
  --hf_grpo models/content/radiology-grpo/checkpoints/grpo_vision \
  --pt_baseline checkpoints/baseline/baseline_best.pt \
  --pt_grpo checkpoints/grpo/grpo_stepXXXX.pt \
  --num_samples 200
```
It reports BLEU, ROUGE-L, and overlap for any provided checkpoints.

Notes
-----
- GRPO trainer: top-p sampling, EOS masking, `[unused*]` blocking, per-image group advantages, KL to baseline.
- Reward: overlap term with zero penalty by default; plug in real CheXbert/RadGraph/BLEU/ROUGE/CIDEr scorers and retune weights for clinical fidelity.
- Torch key_padding_mask warning is harmless; flash attention warning just indicates a non-flash build.

Optional: upstream-style scripts (if you have them available)
------------------------------------------------------------
If you also use the upstream `radiology-grpo` scripts (not bundled here), the typical workflow is:
- Dataset prep:
  ```
  python scripts/convert_iu_to_jsonl_with_images.py
  python scripts/test_image_dataset.py
  ```
- Training:
  ```
  python -m src.train_supervised_vision
  python -m src.train_grpo_vision
  ```
- Testing / metrics (BLEU/ROUGE/etc.):
  ```
  # Supervised baseline
  python scripts/eval_vision_metrics.py --model_ckpt checkpoints/supervised_vision --num_samples 200
  # GRPO-tuned model
  python scripts/eval_vision_metrics.py --model_ckpt checkpoints/grpo_vision --num_samples 200
  ```
These scripts are from the upstream repo; ensure you have them and their dependencies if you want this path.
