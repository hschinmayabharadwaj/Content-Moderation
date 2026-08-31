"""
CLI Analyzer for the trained content-moderation models in trained_hugging_face_models/.

Loads all 5 phases (text, multilingual, image, OCR, multimodal fusion, HITL tiers)
described by manifest.json / config.json, then scores text or image (or image+text)
input and reports across the known RESOURCE GAPS:

    * Multilingual           -> English / Hindi / Kannada (+ code-mixed / Hinglish)
    * OCR (image text)       -> text embedded in images
    * Multimodal             -> text + image fused decision
    * Defamation             -> personal/name-targeting attack detection
    * Deep contextual reason -> negation / reclamation / quoted-mention handling
                               (beyond naive keyword flagging)
    * Static dataset over-reliance -> detects if the decision is dominated by
                               Reddit/Insta-style lexical markers without context

Usage:
    python model_analyzer.py --text "your sentence"
    python model_analyzer.py --image path/to/image.jpg
    python model_analyzer.py --text "..." --image path/to/image.jpg   (multimodal fusion)
    python model_analyzer.py --lang-check                            (drives gap suite)
    python model_analyzer.py                                         (REPL mode)

Markers: use --flag "defamation,multilingual" to focus on specific gaps, or
--flag all (default) to report every gap and the fused verdict.
"""
import os
import sys
import json
import re
import argparse
import unicodedata
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "trained_hugging_face_models"

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception as e:  # pragma: no cover
    TORCH_OK = False
    _TORCH_ERR = e

try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    HF_OK = True
except Exception as e:  # pragma: no cover
    HF_OK = False
    _HF_ERR = e

try:
    import torchvision
    from torchvision import transforms
    TV_OK = True
except Exception as e:  # pragma: no cover
    TV_OK = False
    _TV_ERR = e

try:
    import numpy as np
    NP_OK = True
except Exception:
    NP_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


# --------------------------------------------------------------------------- #
#  Manifest / model-availability discovery (mirrors manifest.json)
# --------------------------------------------------------------------------- #
def load_manifest():
    mpath = CKPT / "manifest.json"
    if mpath.exists():
        with open(mpath, encoding="utf-8") as f:
            return json.load(f)
    return {}


def read_config(subfolder):
    cfg = CKPT / subfolder / "config.json"
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
#  Model definitions (replicated 1:1 from the training scripts)
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """Phase 1: roberta-base multi-label (6 labels)."""
    def __init__(self, model_name, num_labels, hidden=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.Linear(self.config.hidden_size, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        hidden = self.relu(self.hidden_layer(pooled))
        return self.classifier(self.dropout(hidden))


class MultiClassifier(nn.Module):
    """Phase 2 + Phase 5: xlm-roberta-base binary (toxic/clean)."""
    def __init__(self, model_name, hidden=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.Linear(self.config.hidden_size, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        hidden = self.relu(self.hidden_layer(pooled))
        return self.classifier(self.dropout(hidden)).squeeze(1)


def build_resnet18(num_classes):
    model = torchvision.models.resnet18(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 256), nn.ReLU(),
                             nn.Dropout(0.3), nn.Linear(256, num_classes))
    return model


class OCRModel(nn.Module):
    """Phase 3 OCR: resnet18 encoder + GRU decoder (greedy inference)."""
    CHARS = " abcdefghijklmnopqrstuvwxyz0123456789-.,:;!?'\"%()/+&@"
    MAX_LEN = 40

    def __init__(self, vocab_size=None, hidden=256, embed=128):
        super().__init__()
        vocab_size = vocab_size or (len(self.CHARS) + 1)
        enc = torchvision.models.resnet18(weights=None)
        enc.fc = nn.Identity()
        self.enc = enc
        self.proj = nn.Linear(512, hidden)
        self.embed = nn.Embedding(vocab_size, embed)
        self.gru = nn.GRU(input_size=embed + hidden, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def decode(self, img):
        x = self.enc(img)
        feat = torch.tanh(self.proj(x)).unsqueeze(1)
        inp = torch.zeros(img.size(0), 1, dtype=torch.long, device=img.device)
        h = None
        cur = feat
        c2i = {c: i + 1 for i, c in enumerate(self.CHARS)}
        i2c = {v: k for k, v in c2i.items()}
        out_tokens = []
        for _ in range(self.MAX_LEN):
            emb = self.embed(inp)
            x_in = torch.cat([cur, emb], dim=2)
            out, h = self.gru(x_in, h)
            tok = self.fc(out).argmax(-1).item()
            if tok == 0:
                break
            out_tokens.append(i2c.get(tok, ""))
            inp = torch.tensor([[tok]], device=img.device)
            cur = out
        return "".join(out_tokens).strip()


class ImageEmbedder(nn.Module):
    """Phase 4 image side: resnet18 backbone -> 512-d."""
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.fc = nn.Identity()
        self.backbone = base

    def forward(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)


class FusionLayer(nn.Module):
    """Phase 4: late-fusion of text (768) + image (512) -> binary."""
    def __init__(self, text_dim=768, image_dim=512, hidden=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb).unsqueeze(1)
        i = self.image_proj(image_emb).unsqueeze(1)
        att, _ = self.attn(t, i, i)
        att = self.norm(att + t)
        fused = torch.cat([t.squeeze(1), att.squeeze(1)], dim=1)
        return self.classifier(fused).squeeze(1)


# --------------------------------------------------------------------------- #
#  Framework
# --------------------------------------------------------------------------- #
class ModelBundle:
    """Lazy, tolerant loader of every phase model from the trained folder."""
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.manifest = load_manifest()
        self.ready = {}
        self.errors = {}
        self.hitl = {}
        self._load_hitl()
        self._load_phase1()
        self._load_phase2()
        self._load_phase3()
        self._load_ocr()
        self._load_phase4()

    # ---- core capability gate ----
    def _need(self, *flags):
        missing = [f for f in flags if not f]
        if not TORCH_OK:
            return False
        if "text" in flags and not HF_OK:
            return False
        if "image" in flags and not TV_OK:
            return False
        return True

    def _load_hitl(self):
        p = CKPT / "phase5" / "hitl_config.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                self.hitl = json.load(f)
        else:
            self.hitl = None

    def _load_phase1(self):
        sub = "phase1"
        try:
            if not (HF_OK and TORCH_OK):
                self.errors[sub] = "HF/torch unavailable"; return
            cfg = read_config(sub)
            ck = torch.load(CKPT / sub / "best_model.pt", map_location=self.device, weights_only=False)
            labels = cfg.get("labels") or ck.get("labels") or []
            model = TextClassifier("roberta-base", num_labels=len(labels)).to(self.device)
            model.load_state_dict(ck["state_dict"]); model.eval()
            tok = AutoTokenizer.from_pretrained("roberta-base")
            self.ready["phase1"] = {"model": model, "tok": tok, "labels": labels,
                                    "val_acc": cfg.get("best_val_acc")}
        except Exception as e:
            self.errors[sub] = str(e)

    def _load_phase2(self):
        sub = "phase2"
        try:
            if not (HF_OK and TORCH_OK):
                self.errors[sub] = "HF/torch unavailable"; return
            cfg = read_config(sub)
            ck = torch.load(CKPT / sub / "best_model.pt", map_location=self.device, weights_only=False)
            model = MultiClassifier("xlm-roberta-base").to(self.device)
            model.load_state_dict(ck["state_dict"]); model.eval()
            tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.ready["phase2"] = {"model": model, "tok": tok,
                                    "val_acc": cfg.get("best_val_acc"),
                                    "languages": cfg.get("languages", [])}
        except Exception as e:
            self.errors[sub] = str(e)

    def _load_phase3(self):
        sub = "phase3"
        try:
            if not (TV_OK and TORCH_OK):
                self.errors[sub] = "torchvision/torch unavailable"; return
            cfg = read_config(sub)
            ck = torch.load(CKPT / sub / "best_model.pt", map_location=self.device, weights_only=False)
            model = build_resnet18(2).to(self.device)
            model.load_state_dict(ck["state_dict"]); model.eval()
            self.ready["phase3"] = {"model": model, "categories": cfg.get("categories", ["real", "ai"]),
                                    "val_acc": cfg.get("best_val_acc")}
        except Exception as e:
            self.errors[sub] = str(e)

    def _load_ocr(self):
        sub = "phase3_ocr"
        try:
            if not (TV_OK and TORCH_OK):
                self.errors[sub] = "torchvision/torch unavailable"; return
            cfg = read_config(sub)
            vocab = cfg.get("vocab_size", 54)
            ck = torch.load(CKPT / sub / "best_model.pt", map_location=self.device, weights_only=False)
            model = OCRModel(vocab_size=vocab).to(self.device)
            model.load_state_dict(ck["state_dict"]); model.eval()
            self.ready["phase3_ocr"] = {"model": model, "val_acc": cfg.get("best_next_token_acc")}
        except Exception as e:
            self.errors[sub] = str(e)

    def _load_phase4(self):
        sub = "phase4"
        try:
            if not (TV_OK and TORCH_OK and HF_OK):
                self.errors[sub] = "torchvision/torch/HF unavailable"; return
            ck = torch.load(CKPT / sub / "fusion.pt", map_location=self.device, weights_only=False)
            text_emb = AutoModel.from_pretrained("xlm-roberta-base").to(self.device).eval()
            image_emb = ImageEmbedder().to(self.device).eval()
            fusion = FusionLayer().to(self.device).eval()
            image_emb.load_state_dict(ck["image_embed_state"])
            fusion.load_state_dict(ck["fusion_state"])
            tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.ready["phase4"] = {"text_emb": text_emb, "image_emb": image_emb,
                                    "fusion": fusion, "tok": tok,
                                    "val_acc": ck.get("val_acc", read_config(sub).get("best_val_acc"))}
        except Exception as e:
            self.errors[sub] = str(e)

    # ------------------------------------------------------------------ #
    #  phase1: multilabel text
    # ------------------------------------------------------------------ #
    def phase1(self, text):
        if "phase1" not in self.ready:
            return None
        m = self.ready["phase1"]
        enc = m["tok"](text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = m["model"](enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return {lab: float(p) for lab, p in zip(m["labels"], probs)}

    # ------------------------------------------------------------------ #
    #  phase2: binary toxic (multilingual)
    # ------------------------------------------------------------------ #
    def phase2_score(self, text):
        if "phase2" not in self.ready:
            return None
        m = self.ready["phase2"]
        enc = m["tok"](text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logit = m["model"](enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        return float(torch.sigmoid(logit).item())

    # ------------------------------------------------------------------ #
    #  phase3: AI vs real image
    # ------------------------------------------------------------------ #
    def _img_tensor(self, img):
        tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return tf(img.convert("RGB")).unsqueeze(0).to(self.device)

    def phase3(self, img):
        if "phase3" not in self.ready:
            return None
        m = self.ready["phase3"]
        with torch.no_grad():
            prob = torch.softmax(m["model"](self._img_tensor(img)), dim=1).cpu().numpy()[0]
        return {c: float(p) for c, p in zip(m["categories"], prob)}

    # ------------------------------------------------------------------ #
    #  OCR: decode text embedded in image
    # ------------------------------------------------------------------ #
    def ocr_decode(self, img):
        if "phase3_ocr" not in self.ready:
            return None
        m = self.ready["phase3_ocr"]
        with torch.no_grad():
            return m["model"].decode(self._img_tensor(img))

    # ------------------------------------------------------------------ #
    #  phase4: multimodal fusion score
    # ------------------------------------------------------------------ #
    def phase4_score(self, text, img):
        if "phase4" not in self.ready:
            return None
        m = self.ready["phase4"]
        enc = m["tok"](text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            t_out = m["text_emb"](enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            t = t_out.last_hidden_state[:, 0, :]
            i = m["image_emb"](self._img_tensor(img))
            logit = m["fusion"](t, i)
        return float(torch.sigmoid(logit).item())


def decide(score, hitl):
    """Apply Phase 5 HITL tier thresholds to a score in [0,1]."""
    if hitl is None:
        return "AUTO-APPROVE" if score < 0.5 else "HUMAN-REVIEW", score
    tiers = hitl.get("tiers", {})
    app = tiers.get("auto_approve", {}).get("max_score", 0.5)
    rem = tiers.get("auto_remove", {}).get("min_score", 0.6)
    if score <= app:
        return (f"AUTO-APPROVE (score<={app:.2f})", score)
    if score >= rem:
        return (f"AUTO-REMOVE (score>={rem:.2f})", score)
    return (f"HUMAN-REVIEW ({app:.2f}<score<{rem:.2f})", score)


# --------------------------------------------------------------------------- #
#  RESOURCE-GAP ANALYSES  (rule-based probes layered on top of the DNN scores)
# --------------------------------------------------------------------------- #
# Obscene / bad-word lexicon (Reddit/Insta-flavoured, multilingual).
_BAD_WORDS = {
    "en": ["fuck", "fucking", "shit", "bitch", "asshole", "bastard", "cunt",
           "dick", "piss", "slut", "whore", "bullshit", "stupid", "idiot",
           "dumb", "moron", "loser", "scum", "trash", "nazi", "retard",
           "hate", "kill yourself", "die", "worthless"],
    "hi": ["गधा", "कमीना", "मुर्ख", "बेवकूफ", "चूतिया", "मादरचोद", "भोसड़ी",
           "हरामी", "कुत्ता", "साला", "बकवास"],
    "hin": ["gandu", "chutiya", "madarchod", "bhosdi", "harami", "bhadva",
            "kutta", "sala", "bakwas", "bevakoof", "kamina", "gadha"],
    "kn": ["kulla", "punda", "kawtha", "sulli", "gobbara", "doddapakshiga",
           "sulle", "hudgir", "buddhi", "manda", "beakku"],
    "kan": ["ಕುಲ್ಲಾ", "ಪುಂಡ", "ಕತ್ತೆ", "ಮೂರ್ಖ", "ಬಕವಾಸ", "ಹರಾಮಿ"],
}

# Reddit / Instagram-style markers (static-dataset over-reliance probe).
_SOCIAL_MARKERS = ["lol", "lmfao", "omg", "wtf", "af", "tbh", "smh", "ikr",
                   "ngl", "imo", "ily", "srsly", "btw", "rn", "fr", "no cap",
                   "sus", "based", "cringe", "ratio", "pog", "bro", "dude",
                   "yaar", "bhai", "daa", "guru", "mama", "sigma", "simping"]

# Contextual-reasoning lexicon: things a keyword matcher would falsely flag.
_RECLAIMED = ["queer", "slut", "dyke"]
_QUOTED_MENTION = re.compile(r"[“”\"'`]|said|says|reportedly|according|reported|"
                             r"q(\s)?uote|your (post|comment|message)")
_NEGATION = re.compile(r"\b(not|no|never|don'?t|won'?t|can'?t|isn'?t|aren'?t|"
                       r"didn'?t|wasn'?t|weren?t|doesn'?t|ain'?t|unless|without)\b",
                       re.IGNORECASE)


def is_latin(text):
    return bool(re.search(r"[A-Za-z]", text))


def script_distribution(text):
    dist = {}
    for ch in text:
        if not ch.isspace():
            if '\u0900' <= ch <= '\u097F':
                dist["Devanagari(Hindi)"] = dist.get("Devanagari(Hindi)", 0) + 1
            elif '\u0C80' <= ch <= '\u0CFF':
                dist["Kannada"] = dist.get("Kannada", 0) + 1
            elif '\u0030' <= ch <= '\u024F' or ch in ".,!?;:'\"()%/+&@":
                dist["Latin"] = dist.get("Latin", 0) + 1
            else:
                dist[unicodedata.name(ch, "Other")] = dist.get(unicodedata.name(ch, "Other"), 0) + 1
    return dist


def detect_language(text):
    """Heuristic multilingual detection: Kannada / Hindi / English / Hinglish (/ other)."""
    scripts = script_distribution(text)
    has_kn = scripts.get("Kannada", 0) > 0
    has_hi = scripts.get("Devanagari(Hindi)", 0) > 0
    has_lat = scripts.get("Latin", 0) > 0
    if has_kn:
        return "Kannada (kn)"
    if has_hi:
        return "Hindi (hi / Devanagari)"
    if has_lat:
        words = set(re.split(r"\W+", text.lower()))
        roman_kn = ["kulla", "punda", "guru", "maga", "maadthidya", "aitu", "enu",
                    "alla", "ide", "yaake", "hudugaru", "doddavaru", "beku", "illa"]
        if any(w in words for w in roman_kn):
            return "Kannada (Kanglish / romanized)"
        roman_hinglish = ["yaar", "bhai", "nahi", "kya", "ki", "hai", "ho", "mein",
                          "aap", "kar", "tha", "ke", "ka", "bakwas", "gandu", "chutiya",
                          "madarchod", "kamina", "bevakoof"]
        if any(w in words for w in roman_hinglish):
            return "Hinglish / code-mixed"
        return "English (en)"
    return "Unknown/Other"


def find_bad_words(text, lang):
    text_lower = text.lower()
    hits = {key: [] for key in _BAD_WORDS}
    for key, words in _BAD_WORDS.items():
        for w in words:
            if w in text_lower:
                hits[key].append(w)
    return hits


def defamation_analysis(text):
    """Probe for defamation / personal targeting: direct 2nd-person abuse, slur attacks,
    hyperbole accusations that recur in defamatory Reddit/Insta attacks."""
    findings = []
    direct_attack = re.compile(
        r"\b(you|u|tu|tum|nee|ninna|aap)\b.*\b("
        r"idiot|stupid|worthless|disgusting|scum|piece of shit|garbage|trash|"
        r"loser|failure|fraud|fake|liar|cheat|thief|criminal|gandu|charatherless|"
        r"bevakoof|madarchod|buddhi|manda)\b", re.IGNORECASE)
    if direct_attack.search(text):
        findings.append("Direct 2nd-person abusive attack (defamation-adjacent) detected.")
    slur_targeting = re.compile(r"\b(user|member|you|your|people like you|type|"
                                r"kind|community|outsider|migrant|minority)\b", re.I)
    if direct_attack.search(text) and slur_targeting.search(text):
        findings.append("Attack targets the person/group identity (defamation profile).")
    false_accusation = re.compile(r"\b(scam|fraud|fake|liar|corrupt|cheat|steal|"
                                 r"slander|defame|defamation|ruin (your|his|her) reputation)\b", re.I)
    if false_accusation.search(text):
        findings.append("Contains accusation language that could constitute defamation.")
    if not findings:
        findings.append("No defamation / personal-targeting pattern detected.")
    return findings


def context_analysis(text):
    """Deep-contextual-reasoning probe: negation, reclamation, quoted mention."""
    findings = []
    lowered = text.lower()
    reclaimed_hits = [w for w in _RECLAIMED if w in lowered]
    if reclaimed_hits:
        negated = bool(_NEGATION.search(text))
        quoted = bool(_QUOTED_MENTION.search(text))
        reason = "reclaimed/self-usage"
        if negated:
            reason = "used with negation (not/hardly)"
        if quoted or "said" in lowered:
            reason = "reported/quoted mention (e.g., quoting a slur/insult)"
        findings.append(f"Nuance flag: '{', '.join(reclaimed_hits)}' appears with {reason}. "
                        f"-> keyword-only model would OVER-flag. Context matters.")
        return findings, "needs_context"
    if _NEGATION.search(text):
        findings.append("Negation present (a keyword-only scorer may mis-bucket this).")
    return findings, "ok"


def social_overreliance(text):
    """Detect if the 'toxicity' signal is dominated by Reddit/Insta-style surface markers."""
    lowered = text.lower()
    marker_hits = [m for m in _SOCIAL_MARKERS if m in lowered]
    if len(marker_hits) >= 2:
        return (f"High static-dataset marker density ({', '.join(marker_hits[:5])}). "
                f"Likely overfit to Reddit/Insta slang; weigh contextual signal, not just lexicons."), True
    return "Marker density low; static-dataset over-reliance less likely here.", False


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self):
        self.lines = []

    def h(self, s, ch="="):
        self.lines.append(f"\n{'#'*4} {s} {'#'*4}")
        self.lines.append(ch * 60)

    def kv(self, k, v):
        self.lines.append(f"  {k:<34}: {v}")

    def add(self, s=""):
        self.lines.append(s)

    def dump(self):
        out = "\n".join(self.lines)
        print(out)
        return out


def score_bar(score, width=28):
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def flag(score, hi=0.5):
    return "🔴" if score >= hi else "🟢"


# --------------------------------------------------------------------------- #
#  Main analyzer entry
# --------------------------------------------------------------------------- #
def analyze(bundle, text, image=None, focus=None):
    focus = focus or ["all"]
    allg = "all" in focus
    rep = Report()

    rep.h("CONTENT-MODERATION ANALYZER  (from trained_hugging_face_models/manifest.json)")
    rep.add(f"  device     : {bundle.device}")
    rep.add(f"  focus gaps : {', '.join(focus)}")

    # ----- model load status -----
    rep.h("MODEL LOAD STATUS")
    order = ["phase1", "phase2", "phase3", "phase3_ocr", "phase4", "phase5"]
    names = {"phase1": "P1 roberta multi-label (obscene/insult/threat...)",
             "phase2": "P2 xlm-roberta multilingual binary",
             "phase3": "P3 resnet18 AI-vs-Real",
             "phase3_ocr": "P3-OCR resnet18-GRU image->text",
             "phase4": "P4 text+image fusion",
             "phase5": "P5 HITL tiers / drift"}
    for k in order:
        if k == "phase5":
            ok = bundle.hitl is not None
            extra = "tiers loaded"
        elif k in bundle.ready:
            ok, extra = True, f"val_acc={bundle.ready[k].get('val_acc')!s} (best)" if bundle.ready[k].get("val_acc") else "loaded"
        else:
            ok, extra = False, f"NOT LOADED -> {bundle.errors.get(k,'missing deps')}"
        rep.kv(f"{k:10s}{names[k]}", ("✅ " if ok else "❌ ") + str(extra))
    rep.add("  note: DNN weights need torchvision+torch+HF available & HF checkpoints downloaded.")

    if not text and image is None:
        rep.add("\n  No input supplied. Use --text and/or --image.")
        return rep.dump()

    # ----- language  (multilingual gap) -----
    if allg or "multilingual" in focus:
        lang = "n/a"
        if text:
            lang = detect_language(text)
            rep.h("GAP: MULTILINGUAL  (Kannada / Hindi / English / code-mixed)")
            rep.kv("detected language", lang)
            dist = script_distribution(text)
            rep.kv("script mix", ", ".join(f"{k}={v}" for k, v in dist.items()) or "(empty)")
            p2 = bundle.phase2_score(text)
            if p2 is not None:
                rep.add(f"  multilingual toxic score (P2): {p2:.3f} {score_bar(p2)}  {flag(p2)}")
        if image is not None and text is None:
            rep.h("GAP: MULTILINGUAL")
            rep.add("  (image-only — run OCR below to reveal embedded script)")

    # ----- phase1 multilabel text (obscene/bad-word flagging) -----
    if text and (allg or any(g in focus for g in
                             ["obscene", "text", "defamation", "context", "multilingual"])):
        rep.h("GAP: OBSCENE / BAD-WORD FLAGGING  (Phase 1 multi-label)")
        p1 = bundle.phase1(text)
        if p1:
            for lab, p in p1.items():
                shown = "🔴 FLAG" if p >= 0.5 else "✅"
                rep.add(f"  {lab:<16}{p:.3f} {score_bar(p)}  {shown}")
        else:
            rep.add("  phase1 unavailable")
        # raw lexicon overlap
        hits = find_bad_words(text, None)
        total = sum(len(v) for v in hits.values())
        rep.add(f"  lexicon overlap (en/hin/kan/hi): {total} matched"
                + (f" -> {sorted(set(sum(hits.values(), [])))[:6]}" if total else ""))

    # ----- defamation gap -----
    if text and (allg or "defamation" in focus):
        rep.h("GAP: DEFAMATION  (personal/name-targeted attack)")
        for f in defamation_analysis(text):
            rep.add(f"  · {f}")

    # ----- deep contextual reasoning gap -----
    if text and (allg or "context" in focus):
        rep.h("GAP: DEEP CONTEXTUAL REASONING  (negation/reclamation/quoted mention)")
        findings, mode = context_analysis(text)
        for f in findings:
            rep.add(f"  · {f}")
        p1 = bundle.phase1(text)
        if p1 and mode == "needs_context":
            over = sorted((k for k, v in p1.items() if v >= 0.5))
            rep.add(f"  NOTE: machine flags {over or 'nothing'} but context suggests "
                    f"that may be a FALSE POSITIVE.")

    # ----- static dataset over-reliance gap -----
    if text and (allg or "static" in focus):
        rep.h("GAP: STATIC-DATASET OVER-RELIANCE  (Reddit/Insta slang)")
        msg, risk = social_overreliance(text)
        rep.add(f"  · {msg}")

    # ----- OCR gap (image with embedded text) -----
    if image is not None and (allg or "ocr" in focus):
        rep.h("GAP: OCR  (text embedded in image)")
        ocr_txt = bundle.ocr_decode(image)
        if ocr_txt is None:
            rep.add("  phase3_ocr unavailable; try a plain OCR engine or provide --text.")
            ocr_txt = ""
        rep.kv("decoded text (trained OCR)", ocr_txt or "(no text detected)")
        if ocr_txt:
            p2 = bundle.phase2_score(ocr_txt)
            p1 = bundle.phase1(ocr_txt)
            if p2 is not None:
                rep.add(f"  OCR text toxic score (P2): {p2:.3f} {score_bar(p2)} {flag(p2)}")
            if p1:
                flagged = [k for k, v in p1.items() if v >= 0.5]
                rep.kv("OCR text P1 flags", flagged or "none")

    # ----- AI vs real (phase3) -----
    if image is not None and (allg or "image" in focus):
        rep.h("GAP: IMAGE CLASSIFICATION  (Phase 3 - AI vs Real)")
        p3 = bundle.phase3(image)
        if p3:
            for c, p in p3.items():
                rep.add(f"  {c:<8}{p:.3f} {score_bar(p)}  {'🖼️' if p >= 0.5 else ''}")

    # ----- multimodal fusion (phase4) -----
    if image is not None and text is not None and (allg or "multimodal" in focus):
        rep.h("GAP: MULTIMODAL FUSION  (Phase 4 - text + image)")
        f_score = bundle.phase4_score(text, image)
        if f_score is None:
            rep.add("  phase4 unavailable")
        else:
            rep.add(f"  fused toxic score (P4): {f_score:.3f} {score_bar(f_score)} {flag(f_score)}")

    # ----- FINAL VERDICT (P5 HITL) -----
    rep.h("FINAL HITL DECISION  (Phase 5 tiers)")
    cands = []
    guards = []
    if text is not None:
        p2 = bundle.phase2_score(text)
        if p2 is not None:
            cands.append(("text(P2)", p2))
        hits = find_bad_words(text, None)
        total_lex = sum(len(v) for v in hits.values())
        if total_lex:
            guards.append((f"lexicon({total_lex} word)", min(0.9, 0.5 + 0.08 * total_lex)))
        defam = defamation_analysis(text)
        if any("abusive attack" in f for f in defam):
            guards.append(("defamation-attack", 0.9))
    if image is not None:
        p3c = bundle.phase3(image)
        ocr_t = bundle.ocr_decode(image) if image else ""
        p1_ocr = bundle.phase1(ocr_t) if ocr_t else None
        if p3c:
            cands.append(("image-AI(P3)", p3c.get("ai", 0.5)))
        if p1_ocr:
            worst = max(p1_ocr.values())
            cands.append(("OCR-text(P1)", worst))
    if text is not None and image is not None:
        f4 = bundle.phase4_score(text, image)
        if f4 is not None:
            cands.append(("multimodal(P4)", f4))

    if not cands and not guards:
        rep.add("  No model scores available for a verdict.")
    else:
        worst_label, worst = max(cands, key=lambda x: x[1]) if cands else ("(none)", 0.0)
        effective = max([worst] + [g[1] for g in guards])
        # language-aware guard: if a NON-English obscenity is lexically present but the
        # transformer under-scores it, promote to human-review so it is not auto-approved.
        if text is not None and guards:
            lang = detect_language(text)
            if ("Kannada" in lang or "Hindi" in lang or "Hinglish" in lang) and effective < 0.6:
                effective = max(effective, 0.6)
        rep.add("  decision inputs: " + ", ".join(f"{k}={v:.2f}" for k, v in cands)
                + (("  guards: " + ", ".join(f"{k}={v:.2f}" for k, v in guards)) if guards else ""))
        if guards and effective > worst + 1e-6:
            rep.add(f"  guard-rail raised score: DNN={worst:.3f} -> effective={effective:.3f} "
                    f"(lexicon/defam signal compensates for a model blind-spot)")
        decision, _ = decide(effective, bundle.hitl)
        rep.add(f"  driving signal : {'+'.join(worst_label.split('/'))} (score={worst:.3f}, effective={effective:.3f})")
        rep.add(f"  ▶ RESULT       : {decision}")
        if bundle.hitl:
            db = bundle.hitl.get("drift_baseline", {})
            z = (effective - db.get("mean", 0.5)) / (db.get("std", 0.34) + 1e-9)
            rep.add(f"  drift z-score vs baseline : {z:+.2f} "
                    f"(baseline mean={db.get('mean',0):.2f}, p95={db.get('p95',0):.2f})")
            if z > 1.5:
                rep.add("  ⚠ outlier vs training drift baseline — review needed regardless of tier.")

    return rep.dump()


def build_cmd_parser():
    p = argparse.ArgumentParser(description="Analyze text/image across trained content-moderation gaps")
    p.add_argument("--text", type=str, default=None, help="text to analyze")
    p.add_argument("--image", type=str, default=None, help="path to image to analyze")
    p.add_argument("--flag", type=str, default="all",
                   help="comma-separated gaps: obscene,multilingual,defamation,context,"
                        "static,ocr,image,multimodal,all")
    p.add_argument("--lang-check", action="store_true",
                   help="run the multilingual (Kan/Eng/Hin) language-detection probe and exit")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return p


def main(argv=None):
    args = build_cmd_parser().parse_args(argv)
    if not TORCH_OK:
        print("ERROR: torch not importable. Install with pip install torch torchvision transformers")
        return 2

    bundle = ModelBundle(device=args.device)

    if args.lang_check:
        samples = [
            "You are such a stupid idiot and everyone hates you!",
            "तुम इतना बेवकूफ क्यों हो? यह बकवास है।",
            "Nee kulla guru! Enu maadthidya and bowed.",
            "Aaj main bahut khush hoon yaar, thanks a lot!",
            "Nice article, really informative and well written.",
        ]
        for s in samples:
            lang = detect_language(s)
            p2 = bundle.phase2_score(s)
            print(f"  {lang:<28} p2={p2 if p2 is None else round(p2,3)}  | {s[:42]!r}")
        return 0

    image = None
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"ERROR: image not found: {img_path}")
            return 2
        if not PIL_OK:
            print("ERROR: Pillow not available. pip install Pillow")
            return 2
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"ERROR: cannot open image: {e}")
            return 2

    focus = [x.strip().lower() for x in args.flag.split(",") if x.strip()]
    analyze(bundle, args.text, image, focus)

    # REPL mode if no input
    if args.text is None and not args.image:
        print("\n>>> interactive mode. type text or 'image <path>' or 'quit'.")
        while True:
            try:
                raw = input("mod> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue
            if raw.lower() in ("quit", "exit", "q"):
                break
            if raw.lower().startswith("image "):
                p = raw[6:].strip()
                if Path(p).exists() and PIL_OK:
                    image = Image.open(p)
                    analyze(bundle, None, image, focus)
                else:
                    print(f"  cannot load image: {p}")
                continue
            analyze(bundle, raw, None, focus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
