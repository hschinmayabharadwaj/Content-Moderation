"""
Phase 3: Train resnet18 binary classifier on HF AI-vs-Real image dataset.
Exports to trained_hugging_face_models/phase3 (ai_vs_real task)
"""
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "trained_hugging_face_models" / "phase3"
DATA = ROOT / "hf_training" / "_data" / "images" / "ai_vs_real"
IMG_SIZE = 224
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ImgDataset(Dataset):
    def __init__(self, split, transform):
        self.samples = []
        for cls_, label in (("real", 0), ("ai", 1)):
            d = DATA / split / cls_
            for p in sorted(d.glob("*.jpg")):
                self.samples.append((str(p), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def transforms_for(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 256), nn.ReLU(),
                             nn.Dropout(0.3), nn.Linear(256, 2))
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[P3] device={device}")
    train_ds = ImgDataset("train", transforms_for("train"))
    val_ds = ImgDataset("val", transforms_for("val"))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)
    print(f"[P3] train={len(train_ds)} val={len(val_ds)}")

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
            loss = crit(model(img), lbl); loss.backward(); optimizer.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                correct += (model(img).argmax(1) == lbl).sum().item(); total += lbl.size(0)
        acc = correct / total
        print(f"[P3] epoch {epoch+1}: val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "backbone": "resnet18",
                        "categories": ["real", "ai"], "val_acc": acc}, OUT / "best_model.pt")
    elapsed = time.time() - start
    print(f"[P3] DONE in {elapsed/60:.1f} min, best={best:.3f}")
    with open(OUT / "config.json", "w") as f:
        json.dump({"dataset": "ai_vs_real", "backbone": "resnet18",
                   "categories": ["real", "ai"], "best_val_acc": float(best),
                   "train_time_min": round(elapsed/60, 2),
                   "data": "HF AI-vs-Real"}, f, indent=2)


if __name__ == "__main__":
    train()
