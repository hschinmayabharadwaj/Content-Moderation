"""
Phase 4: Train multimodal late-fusion layer.
Combines text embedding (phase2 backbone) + image embedding (phase3 backbone)
into a single fused toxicity score via an attention fusion layer.
Exports to content_moderation_trained/phase4
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "content_moderation_trained" / "phase4"

TEXT_MODEL = "xlm-roberta-base"


class FusionLayer(nn.Module):
    """Late-fusion cross-attention layer: text_emb + image_emb -> fused score"""

    def __init__(self, text_dim=768, image_dim=512, hidden=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        # Cross attention (learnable query from text attending to image, and vice versa)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb).unsqueeze(1)    # [B,1,H]
        i = self.image_proj(image_emb).unsqueeze(1)  # [B,1,H]
        att, _ = self.attn(t, i, i)                  # text attends to image
        att = self.norm(att + t)
        fused = torch.cat([t.squeeze(1), att.squeeze(1)], dim=1)
        return self.classifier(fused).squeeze(1)


class TextEmbedder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # [B, 768]


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18()
        base.fc = nn.Identity()
        self.backbone = base

    def forward(self, x):
        # resnet18 features after avgpool -> [B, 512]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x); x = self.backbone.layer2(x)
        x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


class FusionDataset(Dataset):
    def __init__(self, df, tok, img_transform, max_len=128):
        self.rows = df.to_dict("records")
        self.tok = tok
        self.img_transform = img_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        img = None
        if r.get("image_path") and Path(r["image_path"]).exists():
            img = self.img_transform(Image.open(r["image_path"]).convert("RGB"))
        else:
            img = torch.zeros(3, 224, 224)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img,
            "label": torch.tensor([float(r["toxic"])]),
        }


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[P4] device={device}")

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_df = pd.read_csv(ROOT / "content_moderation_dataset" / "phase4_train.csv")
    # val: reuse phase2 val paired with random images
    val_df = pd.read_csv(ROOT / "content_moderation_dataset" / "phase2_val.csv")
    val_pairs = []
    for i in range(len(val_df)):
        row = val_df.iloc[i]
        toxic = int(row["label"])
        dset = np.random.choice(["nsfw", "hate_symbols", "violence"])
        cls_ = "toxic" if toxic else "safe"
        split = "val"
        files = list((ROOT / "content_moderation_dataset" / "images" / dset / split / cls_).glob("*.png"))
        img_path = str(np.random.choice(files)) if files else ""
        val_pairs.append({"text": row["text"], "toxic": toxic,
                          "image_dset": dset, "image_path": img_path})
    val_df = pd.DataFrame(val_pairs)

    train_ds = FusionDataset(train_df, tok, img_transform)
    val_ds = FusionDataset(val_df, tok, img_transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=2)

    embedder_text = TextEmbedder(TEXT_MODEL).to(device)
    embedder_image = ImageEmbedder().to(device)
    fusion = FusionLayer(text_dim=768, image_dim=512).to(device)

    # Freeze heavy text backbone (use it only as feature extractor for speed)
    for p in embedder_text.backbone.parameters():
        p.requires_grad = False

    params = list(embedder_image.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    epochs = 3

    best = 0.0
    start = time.time()
    embedder_text.eval()
    for epoch in range(epochs):
        embedder_image.train(); fusion.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            img = batch["image"].to(device)
            lbl = batch["label"].to(device).squeeze(1)
            optimizer.zero_grad()
            with torch.no_grad():
                text_emb = embedder_text(input_ids, mask)
            image_emb = embedder_image(img)
            logits = fusion(text_emb, image_emb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, lbl)
            loss.backward()
            optimizer.step()

        embedder_image.eval(); fusion.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                img = batch["image"].to(device)
                lbl = batch["label"].to(device).squeeze(1)
                text_emb = embedder_text(input_ids, mask)
                image_emb = embedder_image(img)
                logits = fusion(text_emb, image_emb)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)
        acc = correct / total
        print(f"[P4] epoch {epoch+1}: val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save({"fusion_state": fusion.state_dict(),
                        "image_embed_state": embedder_image.state_dict(),
                        "text_model": TEXT_MODEL, "val_acc": acc},
                       OUT / "fusion.pt")

    elapsed = time.time() - start
    print(f"[P4] DONE in {elapsed/60:.1f} min, best={best:.3f}")
    with open(OUT / "config.json", "w") as f:
        json.dump({"text_model": TEXT_MODEL, "text_dim": 768, "image_dim": 512,
                   "hidden": 256, "best_val_acc": float(best),
                   "train_time_min": round(elapsed / 60, 2)}, f, indent=2)


if __name__ == "__main__":
    train()
