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

---


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

---

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
  - `indiana/iu_train.csv`, `indiana/iu_val.csv`: 70/10/20 split; `image_path` is relative to `data/`, `report_text` combines Findings/Impression.
  - `mimic-cxr/`: placeholder if you switch to MIMIC-CXR.
- `preprocess/`: dataset loader, image transforms, section extraction, tokenization.
- `models/`: vision encoder, text decoder, baseline wrapper, GRPO trainer.
- `radiology_grpo`: training and testing pipeline for supervised baseline and GRPO model.
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

UI
--
```
python ui/app.py
```
Two ways to use the UI:
- Default (HF checkpoints): UI auto-loads `models/content/radiology-grpo/checkpoints/{supervised_vision,grpo_vision}`. Upload an image; reference text is pulled from `indiana_reports.csv`/`indiana_projections.csv` by filename match, and GRPO falls back to baseline if missing.
- If you prefer your own trained .pt checkpoints (baseline/grpo in `checkpoints/`), adjust the UI loader to point to those paths instead of the HF folders.

Dataset and Model Training
------------------------------------------------------------
All the codes for this are present in the submodule `radiology-grpo`. You would need to add your Kaggle auth token to download the Indiana dataset.
- Installation
  ```
  cd radiology-grpo
  pip install -r requirements.txt
  ```
- Dataset prep:
  ```
  python scripts/convert_iu_to_jsonl_with_images.py
  python scripts/test_image_dataset.py
  ```
- Training:
  ```
  # train the supervised baseline model
  python -m src.train_supervised_vision
  # train the GRPO enhanced model
  python -m src.train_grpo_vision
  ```
- Expected checkpoints:
  ```
  Baseline: `checkpoints/supervised_vision`
  GRPO: `checkpoints/grpo_vision` (latest used in UI)
  ```
- Testing / metrics (BLEU/ROUGE/etc.):
  ```
  # Supervised baseline
  python scripts/eval_vision_metrics.py --model_ckpt checkpoints/supervised_vision --num_samples 791
  # GRPO-tuned model
  python scripts/eval_vision_metrics.py --model_ckpt checkpoints/grpo_vision --num_samples 791
  ```
Dataset can be found at this link: [data](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university).
We have uploaded our trained models on the drive which can be downloaded through this link: [models](https://drive.google.com/drive/folders/1PgDQ4S_4lRJz5E46bwZ9-44Rt05S0dhz?usp=drive_link). For the demo we have used the one that gave us the best result. It has been zipped in `meds_models.zip` in the same drive.
## DEMO
Link to our demo: [Youtube](https://youtu.be/AgQ9w66jz8E)

![Interface](https://github.com/ASMYYY/MedRel/blob/main/images/ui_output.jpeg)
