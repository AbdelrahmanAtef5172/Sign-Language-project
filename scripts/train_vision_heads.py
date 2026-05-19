"""
Train gender and emotion classification heads on frozen ViT-Small backbone.
Extracts 384-dim features once, then trains linear heads via SGD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights/vision")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
GENDER_HEAD_PATH = WEIGHTS_DIR / "gender_head.pt"
EMOTION_HEAD_PATH = WEIGHTS_DIR / "emotion_head.pt"

# ── backbone ────────────────────────────────────────────────────────────────
print(f"Loading ViT-Small backbone on {DEVICE} ...")
from transformers import ViTModel, ViTImageProcessor
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
backbone = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
hf_logging.set_verbosity_warning()
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
backbone.to(DEVICE)
processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
print("Backbone loaded.")


# ── feature extraction helper ──────────────────────────────────────────────
@torch.no_grad()
def extract_features(images, batch_size=BATCH_SIZE):
    """Extract 384-dim features from a list of PIL Images."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        # Ensure all are RGB
        rgb = [im.convert("RGB") if im.mode != "RGB" else im for im in batch]
        inputs = processor(images=rgb, return_tensors="pt", do_rescale=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = backbone(**inputs, interpolate_pos_encoding=True)
        feats.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(feats, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 1. GENDER HEAD TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING GENDER HEAD")
print("="*60)

print("Loading UTK-Face-Revised ...")
gender_ds = load_dataset("deedax/UTK-Face-Revised", split="train")
gender_val_ds = load_dataset("deedax/UTK-Face-Revised", split="valid")

g_images = [gender_ds[i]["image"] for i in range(len(gender_ds))]
g_labels = torch.tensor([1 if gender_ds[i]["gender"] == "Male" else 0 for i in range(len(gender_ds))])

gv_images = [gender_val_ds[i]["image"] for i in range(len(gender_val_ds))]
gv_labels = torch.tensor([1 if gender_val_ds[i]["gender"] == "Male" else 0 for i in range(len(gender_val_ds))])

print(f"Train: {len(g_images)}, Val: {len(gv_images)}")
print("Extracting features for gender ...")
g_feats = extract_features(g_images)
gv_feats = extract_features(gv_images)
print(f"Features shape: {g_feats.shape}")

gender_head = nn.Linear(384, 2).to(DEVICE)
opt_g = torch.optim.Adam(gender_head.parameters(), lr=LR)

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    gender_head.train()
    perm = torch.randperm(len(g_feats))
    total, correct, total_loss = 0, 0, 0.0
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = g_feats[idx].to(DEVICE)
        y = g_labels[idx].to(DEVICE)
        logits = gender_head(x)
        loss = F.cross_entropy(logits, y)
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
        total += len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * len(y)
    train_acc = correct / total
    gender_head.eval()
    with torch.no_grad():
        gv_logits = gender_head(gv_feats.to(DEVICE))
        val_acc = (gv_logits.argmax(1) == gv_labels.to(DEVICE)).float().mean().item()
    print(f"  Epoch {epoch:2d} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"weight": gender_head.weight.data.cpu(), "bias": gender_head.bias.data.cpu()}, GENDER_HEAD_PATH)
        print(f"  -> Saved best (val acc {val_acc:.4f})")

print(f"Gender training done. Best val acc: {best_acc:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. EMOTION HEAD TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING EMOTION HEAD (Sad=0, Neutral=1, Happy=2)")
print("="*60)

# Label map: original FER label -> (our_label, name)
# 3=happy(6292), 4=sad(4253), 6=neutral(4338) → keep
# 0=angry, 1=disgust, 2=fear, 5=surprise → filter out
EMOTION_MAP = {3: 2, 4: 0, 6: 1}  # orig -> our
EMOTION_NAMES = {3: "happy", 4: "sad", 6: "neutral"}

def filter_emotion(ds):
    xs, ys = [], []
    for i in range(len(ds)):
        label = ds[i]["emotion"]
        if label in EMOTION_MAP:
            xs.append(ds[i]["image"])
            ys.append(EMOTION_MAP[label])
    return xs, torch.tensor(ys)

print("Loading FER2013-Enhanced ...")
emo_train = load_dataset("abhilash88/fer2013-enhanced", split="train")
emo_val = load_dataset("abhilash88/fer2013-enhanced", split="validation")

print("Filtering to Sad/Neutral/Happy ...")
e_images, e_labels = filter_emotion(emo_train)
ev_images, ev_labels = filter_emotion(emo_val)
print(f"Train: {len(e_images)}, Val: {len(ev_images)}")

# Class-balanced for the 3-class subset
cls_counts = e_labels.bincount()
print(f"Class distribution (train): Sad={cls_counts[0]}, Neutral={cls_counts[1]}, Happy={cls_counts[2]}")

print("Extracting features for emotion ...")
e_feats = extract_features(e_images)
ev_feats = extract_features(ev_images)
print(f"Features shape: {e_feats.shape}")

emotion_head = nn.Linear(384, 3).to(DEVICE)
opt_e = torch.optim.Adam(emotion_head.parameters(), lr=LR)

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    emotion_head.train()
    perm = torch.randperm(len(e_feats))
    total, correct, total_loss = 0, 0, 0.0
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = e_feats[idx].to(DEVICE)
        y = e_labels[idx].to(DEVICE)
        logits = emotion_head(x)
        loss = F.cross_entropy(logits, y)
        opt_e.zero_grad()
        loss.backward()
        opt_e.step()
        total += len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * len(y)
    train_acc = correct / total
    emotion_head.eval()
    with torch.no_grad():
        ev_logits = emotion_head(ev_feats.to(DEVICE))
        val_acc = (ev_logits.argmax(1) == ev_labels.to(DEVICE)).float().mean().item()
    print(f"  Epoch {epoch:2d} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"weight": emotion_head.weight.data.cpu(), "bias": emotion_head.bias.data.cpu()}, EMOTION_HEAD_PATH)
        print(f"  -> Saved best (val acc {val_acc:.4f})")

print(f"Emotion training done. Best val acc: {best_acc:.4f}")
print(f"\nHeads saved to:\n  {GENDER_HEAD_PATH}\n  {EMOTION_HEAD_PATH}")
print("Done.")
