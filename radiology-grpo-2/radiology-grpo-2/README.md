# steps

## dataset
```
python scripts/convert_iu_to_jsonl_with_images.py
python scripts/test_image_dataset.py
```

## training
```
python -m src.train_supervised_vision
python -m src.train_grpo_vision
```

## testing
```
python -m src.eval_vision_model
# Supervised baseline
python scripts/eval_vision_metrics.py \
  --model_ckpt checkpoints/supervised_vision \
  --num_samples 200
# GRPO-tuned model
python scripts/eval_vision_metrics.py \
  --model_ckpt checkpoints/grpo_vision \
  --num_samples 200
```