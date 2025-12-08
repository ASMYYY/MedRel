import os
import re
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency
    pydicom = None


def default_image_transform(image_size: int = 224, augment: bool = False) -> Callable:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    eval_tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(train_tfms if augment else eval_tfms)


def _load_dicom(path: str) -> Image.Image:
    if pydicom is None:
        raise RuntimeError("pydicom is required to load DICOM files")
    ds = pydicom.dcmread(path)
    array = ds.pixel_array.astype("float32")
    array -= array.min()
    array /= max(array.max(), 1e-6)
    array = (array * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(array).convert("RGB")


def load_image(path: str) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".dcm", ".dicom"]:
        return _load_dicom(path)
    return Image.open(path).convert("RGB")


def extract_sections(text: str) -> Tuple[str, str]:
    lines = text.splitlines()
    impression, findings = [], []
    current = None
    for ln in lines:
        low = ln.lower().strip()
        if re.match(r"^impression", low):
            current = "impression"
            continue
        if re.match(r"^findings?", low):
            current = "findings"
            continue
        if current == "impression":
            impression.append(ln.strip())
        elif current == "findings":
            findings.append(ln.strip())
    imp_txt = " ".join(impression).strip()
    fin_txt = " ".join(findings).strip()
    if not imp_txt and not fin_txt:
        # fallback: parse inline "Findings: ... Impression: ..." variants
        pattern = re.compile(r"findings\s*:\s*(.*?)(?:impression\s*:\s*(.*))?$", re.IGNORECASE | re.DOTALL)
        m = pattern.search(text)
        if m:
            fin_txt = (m.group(1) or "").strip()
            imp_txt = (m.group(2) or "").strip()
    if not imp_txt and not fin_txt:
        # last resort: treat whole text as findings
        fin_txt = text.strip()
    return imp_txt, fin_txt


class MIMICCXRDataset(Dataset):
    """
    Expects a dataframe with columns:
        - image_path: relative or absolute path to the study image
        - report_path (optional if text provided)
        - impression/findings (optional pre-extracted text columns)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        tokenizer,
        max_length: int = 256,
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = default_image_transform(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def _get_report_text(self, row: pd.Series) -> str:
        if "report_text" in row and isinstance(row["report_text"], str) and row["report_text"].strip():
            return row["report_text"]
        if "report_path" in row and isinstance(row["report_path"], str):
            path = row["report_path"]
            if not os.path.isabs(path):
                path = os.path.join(self.image_root, path)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        if "impression" in row or "findings" in row:
            imp = row.get("impression", "") or ""
            fin = row.get("findings", "") or ""
            return f"Impression: {imp}\nFindings: {fin}"
        return ""

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        img = load_image(image_path)
        img = self.image_transform(img)

        report_text = self._get_report_text(row)
        impression, findings = extract_sections(report_text)
        section_text = f"Impression: {impression} Findings: {findings}".strip()
        tokenized = self._tokenize(section_text)

        return {
            "image": img,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "raw_report": report_text,
            "impression": impression,
            "findings": findings,
        }


def collate_reports(
    batch: List[Dict[str, torch.Tensor]],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    masks = [item["attention_mask"] for item in batch]
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": masks},
        padding=True,
        return_tensors="pt",
    )
    images = torch.stack([item["image"] for item in batch])
    return {
        "images": images,
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "raw_report": [item["raw_report"] for item in batch],
        "impression": [item["impression"] for item in batch],
        "findings": [item["findings"] for item in batch],
    }


def make_dataloader(
    dataframe: pd.DataFrame,
    image_root: str,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    max_length: int = 256,
    image_size: int = 224,
    augment: bool = False,
) -> DataLoader:
    dataset = MIMICCXRDataset(
        df=dataframe,
        image_root=image_root,
        tokenizer=tokenizer,
        max_length=max_length,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(collate_reports, tokenizer=tokenizer),
    )


def load_split_csvs(
    split_csvs: Dict[str, str],
    image_root: str,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 256,
    image_size: int = 224,
    augment_train: bool = True,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, csv_path in split_csvs.items():
        df = pd.read_csv(csv_path)
        is_train = split.lower() == "train"
        loaders[split] = make_dataloader(
            dataframe=df,
            image_root=image_root,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=is_train,
            max_length=max_length,
            image_size=image_size,
            augment=is_train and augment_train,
        )
    return loaders
