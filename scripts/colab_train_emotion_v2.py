"""
Colab: Train emotion head on FER2013 with heavy augmentation
to generalize to real webcam faces. Frozen backbone.
Produces emotion_head.pt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
import random

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── load FER2013 ─────────────────────────────────────────────────────────────
print("Loading FER2013-enhanced...")
raw = load_dataset("abhilash88/fer2013-enhanced", split="train")
raw_val = load_dataset("abhilash88/fer2013-enhanced", split="validation")
# 3=happy→2, 4=sad→0, 6=neutral→1
EMAP = {3: 2, 4: 0, 6: 1}
emotion_names = ["sad", "neutral", "happy"]

def filter_ds(ds):
    imgs, lbls = [], []
    for i in range(len(ds)):
        lbl = ds[i]["emotion"]
        if lbl in EMAP:
            imgs.append(ds[i]["image"])
            lbls.append(EMAP[lbl])
    return imgs, torch.tensor(lbls)

train_images, train_labels = filter_ds(raw)
val_images, val_labels = filter_ds(raw_val)
cls_counts = train_labels.bincount()
print(f"Train: sad={cls_counts[0]}, neutral={cls_counts[1]}, happy={cls_counts[2]}")
print(f"Val size: {len(val_images)}")

class_weights = (1.0 / cls_counts.float())
class_weights = class_weights / class_weights.sum() * len(cls_counts)
class_weights = class_weights.to(DEVICE)

# ── processor + backbone (FROZEN) ────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
backbone = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
backbone.to(DEVICE)
print("Backbone frozen.")

# ── augmentation ──────────────────────────────────────────────────────────────
def augment(img: Image.Image) -> Image.Image:
    """Heavy augmentation to make FER2013 look more like real webcam crops."""
    img = img.convert("RGB")
    w, h = img.size
    # Random resize + crop (simulate different face scales)
    scale = random.uniform(0.8, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)  # back to original size
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    # Color jitter (simulate different lighting)
    if random.random() > 0.3:
        import torchvision.transforms as T
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        img = jitter(img)
    # Random grayscale (some webcam feeds are B/W-ish)
    if random.random() > 0.8:
        img = img.convert("L").convert("RGB")
    # Gaussian blur (simulate out-of-focus webcam)
    if random.random() > 0.7:
        import torchvision.transforms as T
        blur = T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
        img = blur(img)
    return img

# ── Dataset ──────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if not isinstance(img, Image.Image):
            arr = np.array(img, dtype=np.uint8)
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3:
                img = Image.fromarray(arr)
            else:
                img = Image.fromarray(arr)
        if self.augment:
            img = augment(img)
        else:
            img = img.convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0]
        return pixel_values, self.labels[idx]

train_ds = EmotionDataset(train_images, train_labels, augment=True)
val_ds = EmotionDataset(val_images, val_labels, augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── train linear head ────────────────────────────────────────────────────────
head = nn.Linear(384, 3).to(DEVICE)
optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Pre-extract validation features once
val_feats, val_labels_all = [], []
with torch.no_grad():
    for pixel_values, labels in val_loader:
        pixel_values = pixel_values.to(DEVICE)
        outputs = backbone(pixel_values, interpolate_pos_encoding=True)
        feats = outputs.last_hidden_state[:, 0, :].cpu()
        val_feats.append(feats)
        val_labels_all.append(labels)
val_feats = torch.cat(val_feats).to(DEVICE)
val_labels_all = torch.cat(val_labels_all).to(DEVICE)

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    head.train()
    total, correct, total_loss = 0, 0, 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            outputs = backbone(pixel_values, interpolate_pos_encoding=True)
            features = outputs.last_hidden_state[:, 0, :]
        logits = head(features)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(logits, 1)
        total += len(labels)
        correct += (preds == labels).sum().item()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    scheduler.step()
    train_acc = correct / total
    # Validation
    head.eval()
    with torch.no_grad():
        val_logits = head(val_feats)
        _, preds = torch.max(val_logits, 1)
        val_acc = (preds == val_labels_all).float().mean().item()
    print(f"  Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({"weight": head.weight.data.cpu(), "bias": head.bias.data.cpu()}, "emotion_head.pt")
        print(f"  -> Saved best (val acc {val_acc:.4f})")

print(f"\nBest val acc: {best_val_acc:.4f}")
print("Download emotion_head.pt from Colab → weights/vision/")
