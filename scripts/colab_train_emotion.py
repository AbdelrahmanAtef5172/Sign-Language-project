"""
Colab-ready: Train emotion head + partially fine-tune ViT backbone.
Produces emotion_head.pt — upload back to weights/vision/.
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
from transformers import ViTModel, ViTImageProcessor

# ── config ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── FER+ (cleaned labels) ───────────────────────────────────────────────────
# 0=neutral, 1=happy, 2=sad (FER+ label scheme)
# We need: sad=0, neutral=1, happy=2
# So remap: FER+ 0(neutral)→1, FER+ 1(happy)→2, FER+ 2(sad)→0
FERPLUS_TO_OURS = {0: 1, 1: 2, 2: 0}
KEEP_LABELS = {0, 1, 2}  # only neutral, happy, sad

def filter_ds(ds):
    images, labels = [], []
    for i in range(len(ds)):
        label = ds[i]["label"]
        if label in KEEP_LABELS:
            images.append(ds[i]["image"])
            labels.append(FERPLUS_TO_OURS[label])
    return images, torch.tensor(labels)

print("Loading FERPlus (cleaned labels)...")
# fallback: use FER2013-enhanced if FERPlus unavailable
try:
    train_ds = load_dataset("trpakov/fer-plus", split="train")
    val_ds = load_dataset("trpakov/fer-plus", split="validation")
except Exception:
    print("FERPlus not found, using FER2013-enhanced with stricter filtering...")
    raw = load_dataset("abhilash88/fer2013-enhanced", split="train")
    raw_val = load_dataset("abhilash88/fer2013-enhanced", split="validation")
    # Original FER label → our mapping: 3=happy→2, 4=sad→0, 6=neutral→1
    emap = {3: 2, 4: 0, 6: 1}
    def filter_fer(raw_ds):
        imgs, lbls = [], []
        for i in range(len(raw_ds)):
            lbl = raw_ds[i]["emotion"]
            if lbl in emap:
                imgs.append(raw_ds[i]["image"])
                lbls.append(emap[lbl])
        return imgs, torch.tensor(lbls)
    train_images, train_labels = filter_fer(raw)
    val_images, val_labels = filter_fer(raw_val)

if "train_images" not in dir():
    train_images, train_labels = filter_ds(train_ds)
    val_images, val_labels = filter_ds(val_ds)

cls_counts = train_labels.bincount()
print(f"Train distribution: sad={cls_counts[0]}, neutral={cls_counts[1]}, happy={cls_counts[2]}")
print(f"Train size: {len(train_images)}, Val size: {len(val_images)}")

# ── class weights (inverse frequency) ────────────────────────────────────────
class_weights = (1.0 / cls_counts.float())
class_weights = class_weights / class_weights.sum() * len(cls_counts)
class_weights = class_weights.to(DEVICE)
print(f"Class weights: sad={class_weights[0]:.3f}, neutral={class_weights[1]:.3f}, happy={class_weights[2]:.3f}")

# ── Dataset ──────────────────────────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")

def to_pil(img):
    """Convert image to PIL RGB regardless of input format."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # numpy array or list of pixels
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] in [1, 3, 4]:
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr).convert("RGB")
    # 2D grayscale
    return Image.fromarray(arr, mode="L").convert("RGB")

class EmotionDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = to_pil(self.images[idx])
        import random
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0]
        return pixel_values, self.labels[idx]

train_dataset = EmotionDataset(train_images, train_labels, augment=True)
val_dataset = EmotionDataset(val_images, val_labels, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Model ────────────────────────────────────────────────────────────────────
backbone = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
# Fine-tune last 2 transformer blocks + pooler
for name, param in backbone.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
backbone.to(DEVICE)

emotion_head = nn.Linear(384, 3).to(DEVICE)

# ── Training ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": LR * 0.1},  # lower LR for backbone
    {"params": emotion_head.parameters(), "lr": LR},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    backbone.train()
    emotion_head.train()
    total, correct, total_loss = 0, 0, 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = backbone(pixel_values, interpolate_pos_encoding=True)
        features = outputs.last_hidden_state[:, 0, :]
        logits = emotion_head(features)
        loss = F.cross_entropy(logits, labels, weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(emotion_head.parameters(), 1.0)
        optimizer.step()

        _, preds = torch.max(logits, 1)
        total += len(labels)
        correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    scheduler.step()
    train_acc = correct / total

    # Validation
    backbone.eval()
    emotion_head.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = backbone(pixel_values, interpolate_pos_encoding=True)
            features = outputs.last_hidden_state[:, 0, :]
            logits = emotion_head(features)
            _, preds = torch.max(logits, 1)
            val_correct += (preds == labels).sum().item()
            val_total += len(labels)
    val_acc = val_correct / val_total

    print(f"  Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {"weight": emotion_head.weight.data.cpu(), "bias": emotion_head.bias.data.cpu()},
            "emotion_head.pt"
        )
        print(f"  -> Saved best (val acc {val_acc:.4f})")

print(f"\nBest val acc: {best_val_acc:.4f}")
print("Download emotion_head.pt from Colab files and place in weights/vision/")
