import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import matplotlib.pyplot as plt

from src.dataset import ReportsWithImagesDataset
from src.vision_transforms import default_image_transform

def main():
    jsonl_path = "data/iu_reports_with_images.jsonl"
    ds = ReportsWithImagesDataset(
        jsonl_path=jsonl_path,
        image_transform=default_image_transform(224),
    )

    print("Dataset size:", len(ds))

    sample = ds[0]
    print("Prompt:", sample["prompt"])
    print("Reference:", sample["reference"][:300], "...")
    print("Image tensor shape:", sample["image"].shape)

    # show original image (no transform)
    # reload without transform
    from PIL import Image
    import json

    with open(jsonl_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    obj = json.loads(first_line)
    img_path = obj["image_path"]
    img = Image.open(img_path).convert("RGB")

    plt.imshow(img)
    plt.title("First IU X-ray sample")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
