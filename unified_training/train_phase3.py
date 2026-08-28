"""
Phase 3: Train 3 image classifiers (resnet18) on balanced synthetic data.
nsfw / hate_symbols / violence -> exports to content_moderation_trained/phase3/<dataset>
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "content_moderation_trained" / "phase3"
DATA = ROOT / "content_moderation_dataset" / "images"

IMG_SIZE = 224


class ImgDataset(Dataset):
    def __init__(self, dset, split, transform):
        self.samples = []
        for cls_ in ("safe", "toxic"):
            d = DATA / dset / split / cls_
            for p in sorted(d.glob("*.png")):
                self.samples.append((str(p), 0 if cls_ == "safe" else 1))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def transforms_for(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_feat, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 2),
    )
    return model


def train_one(dset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_out = OUT / dset
    dset_out.mkdir(parents=True, exist_ok=True)
    train_ds = ImgDataset(dset, "train", transforms_for("train"))
    val_ds = ImgDataset(dset, "val", transforms_for("val"))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    epochs = 5

    best = 0.0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = crit(out, lbl)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                correct += (out.argmax(1) == lbl).sum().item()
                total += lbl.size(0)
        acc = correct / total
        print(f"[P3:{dset}] epoch {epoch+1}: val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "backbone": "resnet18",
                        "categories": ["safe", "toxic"], "val_acc": acc},
                       OUT / dset / "best_model.pt")

    elapsed = time.time() - start
    meta = {"dataset": dset, "backbone": "resnet18", "categories": ["safe", "toxic"],
            "best_val_acc": float(best), "train_time_min": round(elapsed / 60, 2)}
    with open(dset_out / "config.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[P3:{dset}] DONE in {elapsed/60:.1f} min, best={best:.3f}")


if __name__ == "__main__":
    for d in ("nsfw", "hate_symbols", "violence"):
        train_one(d)
