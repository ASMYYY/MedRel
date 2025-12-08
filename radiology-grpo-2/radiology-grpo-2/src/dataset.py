import json
from typing import List, Dict
from torch.utils.data import Dataset


class ReportsDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Expect {"prompt": "...", "reference": "..."}
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# src/dataset.py (add below existing ReportsDataset)

import json
from typing import List, Dict
from torch.utils.data import Dataset
from PIL import Image
import torch

def collate_with_images(batch: List[Dict]):
    # images: stack along batch dimension
    images = torch.stack([b["image"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    refs = [b["reference"] for b in batch]
    return {"images": images, "prompts": prompts, "references": refs}

class ReportsWithImagesDataset(Dataset):
    """
    Each line in JSONL:
      {
        "image_path": "/path/to/image.png",
        "prompt": "FINDINGS: ",
        "reference": "Report text..."
      }
    """

    def __init__(self, jsonl_path: str, image_transform=None):
        self.samples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        prompt = sample["prompt"]
        reference = sample["reference"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,          # tensor if transform set
            "prompt": prompt,
            "reference": reference,
        }
