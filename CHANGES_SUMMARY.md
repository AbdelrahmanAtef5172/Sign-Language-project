# Vision Component — Changes from IMPLEMENTATION_GUIDE.md

## Overview

Three categories of changes were made to the Vision component:

1. **Bug fix:** Confidence thresholding was defined in config but never wired into inference
2. **Diagnostic:** Temporarily adjusted smoothing window to test raw model predictions
3. **Attempted fixes** for emotion model failing on real webcam faces (all ultimately reverted/abandoned — see below)

---

## 1. PERMANENT CHANGE — Confidence Threshold Filtering

**File:** `vision/inference/engine.py` — `_update_smoothers_and_get_result()`

**What changed:** The `GENDER_CONFIDENCE_THRESHOLD` (0.7) and `EMOTION_CONFIDENCE_THRESHOLD` (0.6) fields existed in `vision/config.py` but were never read by any code. Predictions with very low confidence (e.g. 0.51 for gender, where the other class gets 0.49) were entered into the temporal smoother, causing rapid flickering between labels.

**Fix:** Before adding a prediction to the smoother, check if its confidence meets the threshold:

```python
if prediction["gender_conf"] >= self.config.GENDER_CONFIDENCE_THRESHOLD:
    self.gender_smoother.add(prediction["gender"], prediction["gender_conf"])
if prediction["emotion_conf"] >= self.config.EMOTION_CONFIDENCE_THRESHOLD:
    self.emotion_smoother.add(prediction["emotion"], prediction["emotion_conf"])
```

If confidence is below threshold, the prediction is silently dropped — the smoother keeps returning its previous stable value. This eliminates the male/female flickering problem.

**Guide impact:** Section 7.4 (Temporal Smoothing) and `_update_smoothers_and_get_result` in the component spec should document this filtering step. The config values were already listed in Section 7.6 (Configuration Constants).

---

## 2. DIAGNOSTIC ONLY (reverted) — Smoothing Window Test

**File:** `vision/config.py`

**What changed:** Temporarily set `GENDER_SMOOTHING_WINDOW = 1` and `EMOTION_SMOOTHING_WINDOW = 1` to bypass temporal smoothing and observe raw per-frame model predictions.

**Why:** To determine whether incorrect emotion predictions were caused by the smoother corrupting output or by the model itself.

**Result:** The model itself was producing incorrect predictions — raw outputs (no smoothing) still showed sad/neutral confusion. **Fully reverted** to original values (30 for gender, 15 for emotion).

---

## 3. ABANDONED — Sad Logit Bias

**File:** `vision/models/vision_model.py` — `predict()`

**What changed:** Subtracted a bias from the sad-class logit before softmax, first 0.4 then 0.2.

```python
emotion_logits[:, 0] -= 0.4  # later changed to 0.2
```

**Why:** The model predicted "sad" for neutral faces. The bias was meant to shift the decision boundary so sad requires stronger evidence.

**Why abandoned:** 0.4 pushed all predictions to happy (too aggressive). 0.2 pushed to sad+happy only (neutral never won). The fundamental problem was the model weights, not the decision boundary.

**Status:** **Fully reverted.** No bias remains in `predict()`.

---

## 4. ABANDONED — ViT-Base + HuggingFace Emotion Model

**File:** `vision/inference/engine.py`

**What changed:** Integrated `mo-thecreator/vit-Facial-Expression-Recognition` (ViT-Base, 85.8M params, trained on AffectNet real-face photos + FER2013 + MMI, 84.3% accuracy) as a separate emotion classifier. The existing ViT-Small model was kept for gender only.

**Why:** FER2013-trained weights (48x48 low-res grayscale) failed to generalize to real webcam face crops. The HF model was trained on real photos and should work on real webcam feeds.

**Why abandoned:** Running two full ViT models per frame (ViT-Small for gender + ViT-Base for emotion) would be ~5x more compute. On CPU this kills real-time performance (~10-15fps instead of 30+). A proper solution would replace ViT-Small entirely with the ViT-Base as a single backbone, but this requires retraining the gender head (different feature dimension: 768 vs 384).

**Status:** **Fully reverted.** No HF model code remains in `engine.py`.

---

## 5. ADDED — Colab Training Scripts (Experimental, not production-tested)

**File:** `scripts/colab_train_emotion.py`

Trains an emotion classification head (Linear(384, 3)) on FER2013-enhanced with:
- FERPlus fallback (cleaned labels)
- ViT-Small backbone partially fine-tuned (last 2 blocks + pooler)
- Class-weighted cross-entropy loss
- Cosine annealing LR schedule
- 30 epochs, gradient clipping

**Result:** Model converged (79% val acc) but predicted "neutral" on all real webcam faces — backbone fine-tuning overfit to FER2013's low-res appearance.

**File:** `scripts/colab_train_emotion_v2.py`

Same task with a different strategy:
- **Frozen** backbone (no fine-tuning — retains general ImageNet features)
- Heavy data augmentation (color jitter, blur, grayscale, rotation, rescale) to bridge FER2013 → real face domain gap
- Pre-extracted validation features for faster evaluation
- 50 epochs, cosine annealing

**Result:** 75% val accuracy. Still failed on real webcam faces. **Root cause confirmed:** FER2013's 48x48 grayscale images are too far from real webcam face crops for any training strategy to bridge the gap with this architecture.

**Guide impact:** These scripts are standalone utilities. They do not affect the inference pipeline and are not called by any code. They exist only for the Colab training workflow.

---

## Root Cause Summary

The emotion prediction failures (neutral predicted as sad, then neutral only, etc.) all trace to one root cause:

**FER2013 is fundamentally unsuitable for real-world deployment.**
- Images are 48x48 grayscale (webcam crops are 224x224 RGB)
- Labels are noisy (human agreement ~65% on sad vs neutral)
- The visual domain gap cannot be overcome with this architecture and dataset

A proper fix requires:
- **Dataset:** AffectNet, RAF-DB, or FERPlus (real face photos with clean labels)
- **Architecture:** Either replace ViT-Small with a face-specialized backbone, or use a pre-trained end-to-end emotion model that was trained on real faces
- **Note:** This would increase model size (~4x) and requires verifying real-time performance on the target hardware

## Files Changed vs IMPLEMENTATION_GUIDE.md

| File | Change | Status |
|------|--------|--------|
| `vision/inference/engine.py` | Confidence threshold filtering in smoother | **Permanent** |
| `vision/config.py` | Temporary smoothing window test | **Reverted** |
| `vision/models/vision_model.py` | Sad logit bias | **Reverted** |
| `vision/inference/engine.py` | HF model integration | **Reverted** |
| `scripts/colab_train_emotion.py` | New training script | **Added (unused at runtime)** |
| `scripts/colab_train_emotion_v2.py` | New training script v2 | **Added (unused at runtime)** |
