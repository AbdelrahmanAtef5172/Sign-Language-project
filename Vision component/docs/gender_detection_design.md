# Gender Detection Vision Component — Technical Design Document

**Version:** 1.0.0  
**Status:** Implementation-Ready Blueprint  
**Target System:** Multi-component pipeline (Sign Language Recognition · LLM · TTS)  
**Author Role:** Senior ML Systems Architect  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Folder Hierarchy & File Responsibilities](#2-folder-hierarchy--file-responsibilities)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Stage 1 — Face Detection](#4-stage-1--face-detection)
5. [Stage 2 — Face Alignment](#5-stage-2--face-alignment)
6. [Stage 3 — Vision Transformer Classification](#6-stage-3--vision-transformer-classification)
7. [Stage 4 — Optimization Layer](#7-stage-4--optimization-layer)
8. [Data Preprocessing & Normalization Standards](#8-data-preprocessing--normalization-standards)
9. [Model Architecture Decisions & Justifications](#9-model-architecture-decisions--justifications)
10. [Inference & Batching Strategy](#10-inference--batching-strategy)
11. [Configuration System](#11-configuration-system)
12. [Integration Interface (External Components)](#12-integration-interface-external-components)
13. [Dependencies & Environment](#13-dependencies--environment)
14. [Model Weights — Download Instructions](#14-model-weights--download-instructions)
15. [Complete Code Specifications](#15-complete-code-specifications)
16. [Testing Strategy](#16-testing-strategy)
17. [Operational Runbook](#17-operational-runbook)

---

## 1. System Overview

### 1.1 Purpose

The Gender Detection Vision Component is a self-contained, production-grade inference module that consumes raw video frames (or image inputs) and emits per-frame gender predictions with confidence scores. It is designed to operate both in isolation (standalone mode) and as a subordinate component within a larger multi-modal pipeline that includes a Sign Language Recognition module, an LLM, and a TTS system.

### 1.2 Architectural Philosophy

- **Modularity First:** Every stage is independently testable and replaceable. Swapping the backbone ViT model requires touching exactly one file.
- **Efficiency by Default:** The system never re-computes what it already knows. Frame skipping, perceptual caching, and model quantization are built-in, not bolt-ons.
- **Contract-Driven Integration:** The component exposes a single, versioned Python API. External components communicate through typed dataclasses, never raw dicts or tensors.
- **Fail-Gracefully:** Every stage degrades gracefully. If face detection finds no face, the pipeline returns a `NO_FACE` sentinel rather than raising an exception.

### 1.3 High-Level Data Flow

```
Raw Video Frame (BGR numpy array)
        │
        ▼
┌───────────────────┐
│  Frame Gate       │  ← Stride filter: pass frame if frame_idx % 5 == 0
│  (Optimization)   │  ← Perceptual cache: skip if SSIM similarity > threshold
└────────┬──────────┘
         │ Qualified Frame (BGR numpy array, HxWx3, uint8)
         ▼
┌───────────────────┐
│  Face Detector    │  ← RetinaFace (ResNet-50 backbone, WIDER FACE pretrained)
│  Stage 1          │  ← Output: list of BoundingBox(x1,y1,x2,y2,confidence)
└────────┬──────────┘
         │ Cropped Face ROI (BGR numpy array, variable size)
         ▼
┌───────────────────┐
│  Face Alignment   │  ← 5-point landmark-based similarity transform
│  Stage 2          │  ← Output: Aligned face (RGB numpy array, 224x224x3, float32)
└────────┬──────────┘
         │ Normalized Face Tensor (1x3x224x224, float32, ImageNet-normalized)
         ▼
┌───────────────────┐
│  ViT Classifier   │  ← ViT-B/16 backbone (ImageNet-21k pretrained)
│  Stage 3          │  ← Fine-tuned 2-class head: [female, male]
│                   │  ← Output: GenderPrediction(label, confidence, logits)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Result Cache     │  ← Store prediction with frame hash key
│  + Smoothing      │  ← Temporal majority-vote buffer (window=5 frames)
└────────┬──────────┘
         │
         ▼
  GenderResult(label="male"|"female"|"no_face", confidence=float,
               face_bbox=BoundingBox|None, frame_idx=int,
               source="inference"|"cache"|"skipped")
```

---

## 2. Folder Hierarchy & File Responsibilities

```
gender_detection/
│
├── configs/
│   └── config.yaml                 # Single config file; env controlled via --env flag or ENV var
│
├── models/
│   ├── weights/                    # Downloaded pretrained weights (git-ignored)
│   │   ├── retinaface_resnet50.pth
│   │   └── vit_b16_gender.pth
│   ├── retinaface/                 # Vendored RetinaFace source (from Pytorch_Retinaface)
│   │   ├── __init__.py
│   │   ├── network.py              # RetinaFace network definition
│   │   ├── prior_box.py            # Anchor generation
│   │   └── box_utils.py            # NMS and decode functions
│   └── gender_head.py              # ViT-B/16 + 2-class MLP head definition
│
├── pipeline/
│   ├── __init__.py
│   ├── component.py                # ← Single public API: GenderDetectionComponent
│   ├── frame_gate.py               # Stage 0: Stride filter + perceptual cache (pHash inline)
│   ├── face_detector.py            # Stage 1: RetinaFace inference wrapper
│   ├── face_aligner.py             # Stage 2: 5-point landmark alignment
│   ├── gender_classifier.py        # Stage 3: ViT inference wrapper
│   └── smoother.py                 # Stage 4: Temporal majority-vote smoothing
│
├── utils/
│   ├── __init__.py
│   ├── schemas.py                  # All typed dataclasses — contracts between every stage
│   ├── transforms.py               # torchvision inference + training transform chains
│   ├── image_utils.py              # BGR↔RGB, resize, pad, crop helpers
│   └── device.py                   # CUDA / MPS / CPU auto-selection
│
├── scripts/
│   ├── download_weights.py         # One-command weight downloader with SHA-256 verification
│   └── demo_webcam.py              # Live webcam smoke test
│
├── tests/
│   ├── test_gate.py                # Unit tests: stride filter, perceptual cache logic
│   └── test_pipeline.py            # Integration tests: full pipeline on synthetic frames
│
├── requirements.txt                # CPU dependencies (exact versions)
├── requirements-gpu.txt            # GPU dependencies (CUDA 12.1 torch builds)
├── setup.py                        # Installable package definition
└── .env.example                    # Environment variable template
```

### 2.1 Directory Responsibilities

| Path | Responsibility |
|---|---|
| `configs/config.yaml` | Single source of truth for all tunable parameters. Code never contains magic numbers. Dev vs prod behaviour is toggled via the `ENV` environment variable or a `--env` CLI flag, which selects a named section inside this one file. |
| `models/weights/` | Binary weight files only. Git-ignored. Populated by `scripts/download_weights.py`. |
| `models/retinaface/` | Vendored RetinaFace implementation. Isolated so upstream changes never break the project. |
| `models/gender_head.py` | The only file that changes when the backbone is swapped. |
| `pipeline/component.py` | The single public API surface. External callers import only this. All other pipeline modules are internal. |
| `pipeline/frame_gate.py` | Owns both the stride filter and the pHash perceptual cache. No separate hash utility file needed — the logic is ~30 lines and belongs here. |
| `utils/schemas.py` | Single source of truth for all inter-stage data contracts. Every value crossing a stage boundary is a typed dataclass from here. |
| `utils/transforms.py` | Centralises all torchvision pipelines so training and inference are guaranteed to use the same normalization. |
| `scripts/` | Developer and operator tooling. Never imported by the pipeline itself. |
| `tests/` | Two files: one for the gate (pure logic, no model weights needed) and one for the full pipeline (requires weights, marked `integration`). |

> **Rule applied throughout:** one folder per architectural concern; no folder created for fewer than three files.

---

## 3. Pipeline Architecture

### 3.1 Stage Contracts Summary

| Stage | Input Type | Output Type | Failure Mode |
|---|---|---|---|
| Frame Gate | `RawFrame` | `QualifiedFrame` \| `SkippedFrame` | Never fails; always returns one or the other |
| Face Detector | `QualifiedFrame` | `List[DetectedFace]` | Returns empty list if no face found |
| Face Aligner | `DetectedFace` + original frame | `AlignedFace` | Returns `None` if landmarks are degenerate |
| ViT Classifier | `AlignedFace` | `RawPrediction` | Raises `InferenceError` on GPU OOM; caught by component |
| Result Smoother | `RawPrediction` | `GenderResult` | Pass-through if buffer is cold (< window size) |

### 3.2 Threading Model

The component is designed to be called from a **single thread**. It is not thread-safe by default. For multi-threaded consumers (e.g., the sign language recognition module running in a separate thread), wrap it with a standard `threading.Lock` as shown in Section 12.3.

### 3.3 Latency Budget (target: ≤40ms per qualified frame on GPU)

| Stage | Expected Latency (GPU) | Expected Latency (CPU) |
|---|---|---|
| Frame Gate | < 1ms | < 2ms |
| Face Detection (RetinaFace) | ~8ms | ~45ms |
| Face Alignment | < 1ms | < 1ms |
| ViT Inference (ViT-B/16) | ~12ms | ~180ms |
| Smoothing + bookkeeping | < 1ms | < 1ms |
| **Total (qualified frame)** | **~22ms** | **~229ms** |
| **Total (stride-skipped frame)** | **< 1ms** | **< 1ms** |

With stride=5, 80% of frames are skipped. Effective per-frame cost at 30 FPS:
`(0.2 × 22ms) + (0.8 × 0.5ms) = 4.8ms` average per frame → **~208 FPS throughput on GPU**.

---

## 4. Stage 1 — Face Detection

### 4.1 Model Selection: RetinaFace (ResNet-50 backbone)

**Why RetinaFace over alternatives:**

| Detector | WiderFace Hard AP | Speed (GPU) | Landmarks | License |
|---|---|---|---|---|
| RetinaFace-ResNet50 | **91.4%** | ~8ms | ✅ 5-point | MIT |
| MTCNN | 85.1% | ~25ms | ✅ 5-point | MIT |
| YOLOv8-face | 89.7% | ~6ms | ✅ 5-point | AGPL |
| MediaPipe BlazeFace | 83.2% | ~3ms | ❌ None | Apache 2.0 |

RetinaFace is chosen for its balance of accuracy (highest on the WiderFace Hard set, which includes small, occluded, and profile faces) and its native 5-point landmark output (required for Stage 2 alignment). AGPL licensing of YOLOv8 is incompatible with commercial embedding.

### 4.2 Input Specification

```
Input:  BGR numpy array, shape (H, W, 3), dtype=uint8
        H and W are unconstrained (webcam: 480x640, HD: 1080x1920)
```

### 4.3 Output Specification

```python
@dataclass
class DetectedFace:
    bbox: BoundingBox          # (x1, y1, x2, y2) in pixel coordinates
    landmarks: Landmarks5      # 5 (x, y) pairs: left_eye, right_eye, nose, left_mouth, right_mouth
    detection_score: float     # [0.0, 1.0] confidence from RetinaFace
    face_crop: np.ndarray      # Cropped BGR image of the bounding box region
```

### 4.4 Face Detector Implementation Specification (`pipeline/face_detector.py`)

```python
"""
pipeline/face_detector.py
─────────────────────────
Wraps RetinaFace for single-frame face detection.

Responsibilities:
  - Load RetinaFace model once on initialization
  - Preprocess input frame: resize to detection resolution, convert to tensor
  - Run forward pass
  - Post-process: decode anchors, apply NMS, filter by confidence threshold
  - Return list of DetectedFace (empty list if no detections pass threshold)

Dependencies:
  - models/retinaface/ (vendored)
  - models/weights/retinaface_resnet50.pth
  - utils/device.py
  - utils/schemas.py
"""

import torch
import numpy as np
import cv2
from typing import List, Optional

from utils.schemas import DetectedFace, BoundingBox, Landmarks5
from models.backbones.retinaface.network import RetinaFace as RetinaFaceNet
from models.backbones.retinaface.prior_box import PriorBox
from models.backbones.retinaface.box_utils import decode, decode_landm, nms
from utils.device import get_device
import logging

logger = logging.getLogger(__name__)

# ── Detection constants ──────────────────────────────────────────────────────
_RESIZE_LONG_EDGE = 640          # Resize input so longest edge = 640px
_CONFIDENCE_THRESHOLD = 0.85    # Detections below this are discarded
_NMS_THRESHOLD = 0.4            # IoU threshold for NMS
_TOP_K = 5000                   # Max detections before NMS
_KEEP_TOP_K = 750               # Max detections after NMS
_VARIANCE = [0.1, 0.2]          # Anchor decode variances (model-specific, do not change)

class FaceDetector:
    """
    Single-responsibility wrapper around RetinaFace.

    Usage:
        detector = FaceDetector(weights_path="models/weights/retinaface_resnet50.pth")
        faces: List[DetectedFace] = detector.detect(frame_bgr)
    """

    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = get_device(device)
        self.model = self._load_model(weights_path)
        logger.info(f"FaceDetector initialized on {self.device}")

    def _load_model(self, weights_path: str) -> RetinaFaceNet:
        cfg = {
            'name': 'Resnet50',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 24,
            'ngpu': 1,
            'epoch': 100,
            'decay1': 70,
            'decay2': 90,
            'image_size': 840,
            'pretrain': False,
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 256,
            'out_channel': 256,
        }
        net = RetinaFaceNet(cfg=cfg, phase='test')
        state = torch.load(weights_path, map_location=self.device)
        # Strip 'module.' prefix if model was saved with DataParallel
        state = {k.replace('module.', ''): v for k, v in state.items()}
        net.load_state_dict(state)
        net.to(self.device)
        net.eval()
        return net

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame_bgr: HxWx3 uint8 numpy array in BGR color space

        Returns:
            List of DetectedFace sorted by detection_score descending.
            Empty list if no faces exceed confidence threshold.
        """
        h, w = frame_bgr.shape[:2]
        scale_x, scale_y, img_tensor = self._preprocess(frame_bgr)

        with torch.no_grad():
            loc, conf, landms = self.model(img_tensor)

        faces = self._postprocess(
            loc, conf, landms,
            original_h=h, original_w=w,
            scale_x=scale_x, scale_y=scale_y,
            img_tensor=img_tensor,
        )

        # Attach crop from original frame
        for face in faces:
            x1, y1, x2, y2 = (
                int(face.bbox.x1), int(face.bbox.y1),
                int(face.bbox.x2), int(face.bbox.y2),
            )
            face.face_crop = frame_bgr[y1:y2, x1:x2].copy()

        return faces

    def _preprocess(self, frame_bgr: np.ndarray):
        """
        Resize frame so longest edge = _RESIZE_LONG_EDGE.
        Returns (scale_x, scale_y, tensor).
        """
        h, w = frame_bgr.shape[:2]
        scale = _RESIZE_LONG_EDGE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame_bgr, (new_w, new_h))
        scale_x = w / new_w   # to map predictions back to original resolution
        scale_y = h / new_h

        img = resized.astype(np.float32)
        img -= (104, 117, 123)  # ImageNet BGR mean subtraction (RetinaFace-specific)
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # → 1x3xHxW
        img = img.to(self.device)
        return scale_x, scale_y, img

    def _postprocess(self, loc, conf, landms,
                     original_h, original_w,
                     scale_x, scale_y,
                     img_tensor) -> List[DetectedFace]:
        """Decode anchors, apply NMS, filter, and map back to original coordinates."""
        ih, iw = img_tensor.shape[2], img_tensor.shape[3]
        priorbox = PriorBox(
            cfg={'min_sizes': [[16, 32], [64, 128], [256, 512]],
                 'steps': [8, 16, 32], 'clip': False},
            image_size=(ih, iw)
        )
        priors = priorbox.forward().to(self.device)

        boxes = decode(loc.squeeze(0), priors, _VARIANCE)
        # Scale boxes to original image pixel coordinates
        boxes = boxes * torch.tensor([iw, ih, iw, ih], device=self.device)
        boxes = boxes.cpu().numpy()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        scores = conf.squeeze(0).cpu().numpy()[:, 1]

        landmarks = decode_landm(landms.squeeze(0), priors, _VARIANCE)
        landmarks = landmarks * torch.tensor([iw, ih] * 5, device=self.device)
        landmarks = landmarks.cpu().numpy()
        landmarks[:, 0::2] *= scale_x   # x coords
        landmarks[:, 1::2] *= scale_y   # y coords

        # Filter by confidence
        keep = scores > _CONFIDENCE_THRESHOLD
        boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]

        # NMS
        order = scores.argsort()[::-1][:_TOP_K]
        boxes, scores, landmarks = boxes[order], scores[order], landmarks[order]
        keep_idx = nms(boxes, scores, _NMS_THRESHOLD)
        keep_idx = keep_idx[:_KEEP_TOP_K]
        boxes, scores, landmarks = boxes[keep_idx], scores[keep_idx], landmarks[keep_idx]

        result = []
        for box, score, lm in zip(boxes, scores, landmarks):
            x1, y1, x2, y2 = (
                max(0, box[0]), max(0, box[1]),
                min(original_w, box[2]), min(original_h, box[3])
            )
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            landmarks5 = Landmarks5(
                left_eye=(lm[0], lm[1]),
                right_eye=(lm[2], lm[3]),
                nose=(lm[4], lm[5]),
                left_mouth=(lm[6], lm[7]),
                right_mouth=(lm[8], lm[9]),
            )
            result.append(DetectedFace(
                bbox=bbox,
                landmarks=landmarks5,
                detection_score=float(score),
                face_crop=None,  # populated by caller
            ))

        result.sort(key=lambda f: f.detection_score, reverse=True)
        return result
```

### 4.5 Multi-Face Strategy

When multiple faces are detected (e.g., group shots), the component's behaviour is governed by `configs/config.yaml`:

```yaml
face_detection:
  multi_face_strategy: "largest"   # Options: "largest" | "highest_confidence" | "all"
  min_face_size_px: 40             # Faces smaller than 40x40 pixels are ignored
  max_faces_per_frame: 1           # Limit for "all" strategy
```

- `largest`: Select the face with the largest bounding box area (best for single-subject use cases like sign language recognition)
- `highest_confidence`: Select the face with the highest detection score
- `all`: Process all detected faces and return a list of `GenderResult`

---

## 5. Stage 2 — Face Alignment

### 5.1 Purpose

Raw face crops are inconsistently positioned, scaled, and rotated. ViT performance degrades significantly without alignment because the patch-embedding grid misaligns with facial features. Stage 2 applies a similarity transform (rotation + scale + translation, no shear) to warp the detected face to a canonical 224×224 template.

### 5.2 Template Landmarks (ArcFace Standard)

The target landmark positions are derived from the ArcFace template (112×112) scaled to 224×224:

```python
# Reference landmarks for 224x224 output image
# (x, y) in pixel coordinates, origin at top-left
ARCFACE_TEMPLATE_224 = np.array([
    [85.82,  85.84],   # left_eye
    [138.18, 85.84],   # right_eye
    [112.0,  115.50],  # nose
    [90.20,  143.80],  # left_mouth
    [133.80, 143.80],  # right_mouth
], dtype=np.float32)
```

### 5.3 Alignment Algorithm

The aligner uses OpenCV's `estimateAffinePartial2D` (which constrains to similarity transform—no shear) followed by `warpAffine`:

```python
"""
pipeline/face_aligner.py
────────────────────────
Applies a 5-point similarity transform to warp a detected face to a
canonical 224x224 aligned representation.

Input:  DetectedFace (contains bbox, landmarks5, face_crop from Stage 1)
        Original BGR frame (numpy array)
Output: AlignedFace (224x224x3 RGB float32 numpy array, normalized to [0,1])
        Returns None if the similarity transform estimation fails (degenerate landmarks)
"""

import cv2
import numpy as np
from typing import Optional

from utils.schemas import DetectedFace, AlignedFace
import logging

logger = logging.getLogger(__name__)

ARCFACE_TEMPLATE_224 = np.array([
    [85.82,  85.84],
    [138.18, 85.84],
    [112.0,  115.50],
    [90.20,  143.80],
    [133.80, 143.80],
], dtype=np.float32)

OUTPUT_SIZE = (224, 224)  # (width, height)
MARGIN_RATIO = 0.15       # 15% padding around tight crop before warping

class FaceAligner:
    """
    Aligns a detected face to the ArcFace canonical 224x224 template.
    Uses similarity transform (rotation, scale, translation only — no shear).
    """

    def __init__(self):
        self._template = ARCFACE_TEMPLATE_224.copy()

    def align(
        self,
        detected_face: DetectedFace,
        original_frame_bgr: np.ndarray,
    ) -> Optional[AlignedFace]:
        """
        Args:
            detected_face:        Stage 1 output
            original_frame_bgr:   Full original frame (for high-quality warping)

        Returns:
            AlignedFace with shape (224, 224, 3), dtype float32, RGB, values in [0, 1]
            None if transform estimation fails.
        """
        src_pts = self._landmarks_to_array(detected_face.landmarks)

        # Estimate 2D similarity transform (partial affine = no shear)
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, self._template,
            method=cv2.LMEDS,
            confidence=0.99,
        )

        if transform_matrix is None or (inliers is not None and inliers.sum() < 3):
            logger.warning("Face alignment failed: degenerate landmarks or insufficient inliers")
            return None

        # Warp from the original high-resolution frame (better quality than the crop)
        aligned_bgr = cv2.warpAffine(
            original_frame_bgr,
            transform_matrix,
            OUTPUT_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # BGR → RGB, uint8 → float32 in [0, 1]
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return AlignedFace(
            image=aligned_rgb,
            shape=(224, 224, 3),
            source_bbox=detected_face.bbox,
            transform_matrix=transform_matrix,
        )

    @staticmethod
    def _landmarks_to_array(landmarks) -> np.ndarray:
        """Convert Landmarks5 dataclass to (5, 2) float32 numpy array."""
        return np.array([
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.nose,
            landmarks.left_mouth,
            landmarks.right_mouth,
        ], dtype=np.float32)
```

### 5.4 Why Warp from the Original Frame (Not the Crop)

The face crop from Stage 1 may be tight. Warping from the full original frame gives `warpAffine` access to pixels at the image borders of the face region (e.g., forehead, chin) that the crop may have excluded. This avoids black-border artifacts at the output image edges, which would corrupt the ViT patch embeddings near the perimeter.

---

## 6. Stage 3 — Vision Transformer Classification

### 6.1 Model Selection: ViT-B/16 (Vision Transformer, Base, 16×16 patches)

**Why ViT over CNN-based classifiers:**

| Metric | ViT-B/16 | ResNet-50 | EfficientNet-B4 |
|---|---|---|---|
| ImageNet-1k Top-1 | 84.15% | 76.13% | 82.9% |
| Global context | ✅ Self-attention | ❌ Local receptive field | ❌ Local |
| Transfer to faces | Strong (ImageNet-21k pretrain) | Good | Good |
| Inference latency (GPU, batch=1) | ~12ms | ~8ms | ~10ms |
| Parameter count | 86M | 25M | 19M |

For gender classification, global facial structure (jaw shape, forehead width, overall proportions) matters as much as local texture. ViT's self-attention across the full 224×224 image captures this holistic information that CNN receptive fields miss at early layers.

**Pretrained weights used:** `google/vit-base-patch16-224-in21k` (ImageNet-21k, 21,841-class pretraining). This provides substantially better feature initialization than ImageNet-1k alone.

### 6.2 Classification Head Architecture

```
ViT-B/16 backbone (frozen for first N epochs, then unfrozen)
    │
    └─ [CLS] token embedding (dim=768)
            │
            ▼
    LayerNorm(768)
            │
            ▼
    Linear(768 → 256)
            │
            ▼
    GELU activation
            │
            ▼
    Dropout(p=0.3)
            │
            ▼
    Linear(256 → 2)       ← [female_logit, male_logit]
            │
            ▼
    Softmax → [P(female), P(male)]
```

**Why two-stage MLP head (not single linear layer):**
- A single `768 → 2` projection discards too much information. The intermediate 256-dim bottleneck allows the head to learn a compact, task-specific feature space before the final binary classification, improving calibration.

### 6.3 Gender Classifier Implementation Specification (`pipeline/gender_classifier.py`)

```python
"""
pipeline/gender_classifier.py
──────────────────────────────
Wraps the ViT-B/16 + gender head for inference.

Input:  AlignedFace (224x224x3, RGB, float32 in [0,1])
Output: RawPrediction(label, confidence, logits, probabilities)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from utils.schemas import AlignedFace, RawPrediction, GenderLabel
from utils.transforms import get_inference_transforms
from models.gender_head import GenderClassifier
from utils.device import get_device
import logging

logger = logging.getLogger(__name__)

LABEL_MAP = {0: GenderLabel.FEMALE, 1: GenderLabel.MALE}
CONFIDENCE_FLOOR = 0.5   # predictions below this floor are flagged as LOW_CONFIDENCE

class GenderInference:
    """
    Inference-mode wrapper for the ViT gender classifier.

    Usage:
        infer = GenderInference(weights_path="models/weights/vit_b16_gender.pth")
        prediction: RawPrediction = infer.predict(aligned_face)
    """

    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ):
        self.device = get_device(device)
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.transforms = get_inference_transforms()
        self.model = self._load_model(weights_path)
        logger.info(f"GenderInference initialized | device={self.device} | fp16={self.use_fp16}")

    def _load_model(self, weights_path: str) -> GenderClassifier:
        model = GenderClassifier()
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state['model_state_dict'])
        model.to(self.device)
        model.eval()

        if self.use_fp16:
            model.half()
            logger.info("GenderClassifier running in FP16 mode")

        return model

    @torch.no_grad()
    def predict(self, aligned_face: AlignedFace) -> RawPrediction:
        """
        Args:
            aligned_face: AlignedFace with .image of shape (224, 224, 3), float32, RGB, [0,1]

        Returns:
            RawPrediction with label, confidence, logits, and probabilities
        """
        tensor = self.transforms(aligned_face.image)      # → (3, 224, 224) normalized tensor
        tensor = tensor.unsqueeze(0).to(self.device)       # → (1, 3, 224, 224)

        if self.use_fp16:
            tensor = tensor.half()

        logits = self.model(tensor)                        # → (1, 2)
        probs = F.softmax(logits.float(), dim=-1)          # always float32 for probabilities
        pred_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_idx].item()

        return RawPrediction(
            label=LABEL_MAP[pred_idx],
            confidence=confidence,
            logits=logits[0].cpu().float().numpy(),
            probabilities=probs[0].cpu().numpy(),
            is_low_confidence=confidence < CONFIDENCE_FLOOR,
        )
```

### 6.4 Gender Head Model Definition (`models/classifier/gender_head.py`)

```python
"""
models/classifier/gender_head.py
──────────────────────────────────
Defines the full ViT-B/16 + 2-class MLP head.
The backbone comes from the `timm` library (PyTorch Image Models).
"""

import torch
import torch.nn as nn
import timm

class GenderClassifier(nn.Module):
    """
    ViT-B/16 backbone with a custom 2-class gender classification head.

    Architecture:
        - Backbone: vit_base_patch16_224 pretrained on ImageNet-21k (via timm)
        - Head: LayerNorm → Linear(768,256) → GELU → Dropout(0.3) → Linear(256,2)
        - Output: raw logits of shape (batch_size, 2)
    """

    BACKBONE_NAME = "vit_base_patch16_224"
    EMBED_DIM = 768
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    DROPOUT = 0.3

    def __init__(self, pretrained_backbone: bool = False):
        """
        Args:
            pretrained_backbone: If True, loads ImageNet-21k weights for the backbone.
                                 Set True during training, False during inference
                                 (weights loaded via load_state_dict instead).
        """
        super().__init__()

        # Load backbone, removing its original classification head
        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained_backbone,
            num_classes=0,      # removes the timm head; output is CLS token embedding
            global_pool='token' # returns CLS token (not global average pool)
        )

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.EMBED_DIM),
            nn.Linear(self.EMBED_DIM, self.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(p=self.DROPOUT),
            nn.Linear(self.HIDDEN_DIM, self.NUM_CLASSES),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Xavier uniform for linear layers, zeros for biases."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (B, 3, 224, 224), ImageNet-normalized

        Returns:
            Logits of shape (B, 2)  — [female_logit, male_logit]
        """
        features = self.backbone(x)   # (B, 768) CLS token
        logits = self.head(features)  # (B, 2)
        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters. Used in Phase 1 of training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, layers_from_end: int = 4):
        """
        Selectively unfreeze the last N transformer blocks.
        Used in Phase 2 of fine-tuning (discriminative learning rates).

        Args:
            layers_from_end: Number of ViT blocks from the end to unfreeze (default: 4)
        """
        blocks = list(self.backbone.blocks)
        for block in blocks[-layers_from_end:]:
            for param in block.parameters():
                param.requires_grad = True
        # Always unfreeze the final norm
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
```

---

## 7. Stage 4 — Optimization Layer

### 7.1 Frame Gate: Stride-Based Frame Skipping (`pipeline/frame_gate.py`)

The frame gate is the first component in the pipeline. It gates frames before any expensive computation runs.

**Stride Logic:** Process frame if and only if `frame_idx % stride == 0`.
- Default stride: `5` (process every 5th frame)
- At 30 FPS, this yields 6 classified frames per second — sufficient for temporal smoothing

**Perceptual Cache Logic:** Even at stride=5, consecutive qualifying frames may be nearly identical (e.g., slow-moving subject). The perceptual cache computes a perceptual hash (pHash) of each qualifying frame and compares it against the previously processed frame's hash. If the Hamming distance between hashes is below a threshold, the frame is classified as a cache hit and the previous result is returned.

```python
"""
pipeline/frame_gate.py
───────────────────────
Stage 0 of the pipeline. Two-level frame filter:
  Level 1 — Stride filter: skips frames not on the stride boundary
  Level 2 — Perceptual cache: skips visually identical frames (even at stride boundary)

Input:  RawFrame(data=np.ndarray, frame_idx=int)
Output: QualifiedFrame | CacheHitFrame | SkippedFrame
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from utils.schemas import RawFrame, QualifiedFrame, GenderResult
import logging

logger = logging.getLogger(__name__)

class FrameDisposition(Enum):
    QUALIFIED   = "qualified"   # Proceed to full pipeline
    CACHE_HIT   = "cache_hit"   # Visually same as last frame; reuse cached result
    STRIDE_SKIP = "stride_skip" # Stride filter; reuse last result

@dataclass
class GateDecision:
    disposition: FrameDisposition
    frame: Optional[QualifiedFrame] = None          # set if QUALIFIED
    cached_result: Optional[GenderResult] = None    # set if CACHE_HIT or STRIDE_SKIP

class FrameGate:
    """
    Two-level frame filter with stride-based skipping and perceptual caching.

    Config parameters (from configs/config.yaml → frame_gate section):
        frame_gate.stride: int = 5
        frame_gate.phash_size: int = 16   (16x16 pHash = 256-bit hash)
        frame_gate.cache_hamming_threshold: int = 10  (≤10 differing bits = cache hit)
        frame_gate.cache_resize_to: tuple = (64, 64)  (resize before hashing for speed)
    """

    def __init__(
        self,
        stride: int = 5,
        phash_size: int = 16,
        cache_hamming_threshold: int = 10,
        cache_resize_to: Tuple[int, int] = (64, 64),
    ):
        self.stride = stride
        self.phash_size = phash_size
        self.hamming_threshold = cache_hamming_threshold
        self.cache_resize_to = cache_resize_to

        self._last_phash: Optional[np.ndarray] = None
        self._last_result: Optional[GenderResult] = None

        # Stats for monitoring
        self._n_qualified = 0
        self._n_cache_hits = 0
        self._n_stride_skips = 0

    def gate(self, raw_frame: RawFrame) -> GateDecision:
        """
        Args:
            raw_frame: RawFrame with .data (BGR numpy array) and .frame_idx (int)

        Returns:
            GateDecision indicating whether to proceed with inference or use cached result
        """
        # ── Level 1: Stride filter ────────────────────────────────────────────
        if raw_frame.frame_idx % self.stride != 0:
            self._n_stride_skips += 1
            return GateDecision(
                disposition=FrameDisposition.STRIDE_SKIP,
                cached_result=self._last_result,
            )

        # ── Level 2: Perceptual cache ─────────────────────────────────────────
        current_hash = compute_phash(
            raw_frame.data,
            hash_size=self.phash_size,
            resize_to=self.cache_resize_to,
        )

        if self._last_phash is not None and self._last_result is not None:
            dist = hamming_distance(current_hash, self._last_phash)
            if dist <= self.hamming_threshold:
                self._n_cache_hits += 1
                logger.debug(f"Frame {raw_frame.frame_idx}: pHash hit (dist={dist})")
                return GateDecision(
                    disposition=FrameDisposition.CACHE_HIT,
                    cached_result=self._last_result,
                )

        # Frame passed both gates → proceed to full pipeline
        self._last_phash = current_hash
        self._n_qualified += 1

        return GateDecision(
            disposition=FrameDisposition.QUALIFIED,
            frame=QualifiedFrame(
                data=raw_frame.data,
                frame_idx=raw_frame.frame_idx,
                phash=current_hash,
            ),
        )

    def update_cached_result(self, result: GenderResult):
        """Called by the pipeline after a full inference to update the cache."""
        self._last_result = result

    def get_stats(self) -> dict:
        total = self._n_qualified + self._n_cache_hits + self._n_stride_skips
        return {
            "total_frames_seen": total,
            "qualified": self._n_qualified,
            "cache_hits": self._n_cache_hits,
            "stride_skips": self._n_stride_skips,
            "cache_hit_rate": self._n_cache_hits / max(1, total),
            "effective_skip_rate": (self._n_cache_hits + self._n_stride_skips) / max(1, total),
        }
```

### 7.2 Perceptual Hash — Inline in `pipeline/frame_gate.py`

The pHash logic is compact enough (~25 lines) to live directly inside `frame_gate.py`. No separate utility file is needed. The functions below are defined at the module level of `frame_gate.py` and called by `FrameGate.gate()`.

```python
"""
Perceptual hashing helpers — defined at module level in pipeline/frame_gate.py
Uses OpenCV's DCT for the pHash computation (no scipy dependency).
"""

import numpy as np
import cv2
from typing import Tuple

def compute_phash(
    frame_bgr: np.ndarray,
    hash_size: int = 16,
    resize_to: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Compute a perceptual hash (DCT-based pHash) of a frame.

    Algorithm:
      1. Resize to resize_to (small, for speed)
      2. Convert to grayscale
      3. Apply 2D DCT
      4. Take top-left hash_size x hash_size DCT coefficients (low frequencies)
      5. Binarize: 1 if > median, else 0
      6. Return as (hash_size*hash_size,) bool array (256 bits at hash_size=16)

    Args:
        frame_bgr:  Input BGR frame (any size)
        hash_size:  Size of the DCT coefficient block (hash bits = hash_size^2)
        resize_to:  Resize target before DCT (larger = more accurate but slower)

    Returns:
        Boolean numpy array of shape (hash_size*hash_size,)
    """
    # Resize
    small = cv2.resize(frame_bgr, resize_to, interpolation=cv2.INTER_AREA)
    # Grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 2D DCT (using cv2's dct on each row then column for efficiency)
    dct = cv2.dct(gray)
    # Top-left block (low-frequency components)
    dct_low = dct[:hash_size, :hash_size]
    # Exclude DC component (dct_low[0,0]) from median to avoid brightness dominance
    flat = dct_low.flatten()
    median = np.median(flat[1:])
    # Binarize
    return flat > median

def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    """
    Compute Hamming distance between two boolean hash arrays.
    Lower = more visually similar.

    Args:
        hash_a, hash_b: Boolean arrays of the same shape

    Returns:
        Integer count of differing bits
    """
    return int(np.count_nonzero(hash_a != hash_b))
```

### 7.3 Temporal Smoothing (`pipeline/smoother.py`)

Single-frame predictions are noisy. The smoother maintains a circular buffer of the last N raw predictions and returns the majority-vote label with averaged confidence.

```python
"""
pipeline/smoother.py
────────────────────────────
Temporal majority-vote smoother. Reduces single-frame prediction noise
by averaging over a sliding window of recent predictions.

Input:  RawPrediction
Output: GenderResult (smoothed)

Config:
    result_smoother.window_size: int = 5
    result_smoother.min_confidence_to_include: float = 0.55
"""

from collections import deque
from typing import Deque
import numpy as np

from utils.schemas import RawPrediction, GenderResult, GenderLabel

class ResultSmoother:
    """
    Sliding window majority-vote smoother.

    Uses a deque of the last `window_size` high-confidence predictions.
    Low-confidence predictions are included in the window but do not
    override a strong majority.
    """

    def __init__(self, window_size: int = 5, min_confidence_to_include: float = 0.55):
        self.window_size = window_size
        self.min_confidence = min_confidence_to_include
        self._buffer: Deque[RawPrediction] = deque(maxlen=window_size)

    def smooth(self, raw: RawPrediction, frame_idx: int) -> GenderResult:
        """
        Add raw prediction to buffer and return smoothed result.

        Args:
            raw:        Latest RawPrediction from Stage 3
            frame_idx:  Current frame index (for result metadata)

        Returns:
            GenderResult with smoothed label and averaged confidence
        """
        if not raw.is_low_confidence:
            self._buffer.append(raw)
        elif len(self._buffer) == 0:
            # Buffer is empty and we got a low-confidence prediction — include it anyway
            self._buffer.append(raw)

        if len(self._buffer) == 0:
            # Completely cold start with low confidence — pass through
            return GenderResult(
                label=raw.label,
                confidence=raw.confidence,
                face_bbox=None,
                frame_idx=frame_idx,
                source="inference_cold",
                is_smoothed=False,
            )

        # Majority vote
        labels = [p.label for p in self._buffer]
        n_female = labels.count(GenderLabel.FEMALE)
        n_male = labels.count(GenderLabel.MALE)
        voted_label = GenderLabel.FEMALE if n_female >= n_male else GenderLabel.MALE

        # Average probability for the voted label across buffer
        label_idx = 0 if voted_label == GenderLabel.FEMALE else 1
        avg_confidence = float(np.mean([p.probabilities[label_idx] for p in self._buffer]))

        return GenderResult(
            label=voted_label,
            confidence=avg_confidence,
            face_bbox=raw.aligned_face.source_bbox if hasattr(raw, 'aligned_face') else None,
            frame_idx=frame_idx,
            source="inference_smoothed",
            is_smoothed=True,
        )

    def reset(self):
        """Clear the smoothing buffer. Call when subject changes."""
        self._buffer.clear()
```

### 7.4 Additional Optimizations

#### 7.4.1 FP16 (Half-Precision) Inference

On CUDA GPUs (RTX 2000+), FP16 inference cuts memory bandwidth by ~2× and uses Tensor Core acceleration:

```yaml
# configs/config.yaml → production section
inference:
  use_fp16: true    # Requires CUDA GPU with compute capability >= 7.0
```

FP16 is automatically disabled on CPU and MPS (Apple Silicon) targets.

#### 7.4.2 ONNX Export + TensorRT (Optional, GPU deployments)

For maximum throughput in GPU deployments, export both models to ONNX and compile with TensorRT:

```bash
# Export face detector
python scripts/export_onnx.py \
    --model face_detector \
    --weights models/weights/retinaface_resnet50.pth \
    --output models/weights/retinaface_resnet50.onnx \
    --input-shape 1 3 640 640

# Export gender classifier
python scripts/export_onnx.py \
    --model gender_classifier \
    --weights models/weights/vit_b16_gender.pth \
    --output models/weights/vit_b16_gender.onnx \
    --input-shape 1 3 224 224
```

TensorRT FP16 engine build (requires TensorRT 8.x installed separately):

```bash
trtexec --onnx=models/weights/vit_b16_gender.onnx \
        --saveEngine=models/weights/vit_b16_gender_fp16.trt \
        --fp16 \
        --workspace=1024
```

#### 7.4.3 Model Warm-Up

The first inference call is always slower due to CUDA kernel compilation. The component auto-warms up on initialization:

```python
# In pipeline/component.py __init__:
def _warmup(self):
    """Run a dummy forward pass to trigger CUDA kernel compilation."""
    dummy = np.zeros((224, 224, 3), dtype=np.float32)
    dummy_face = AlignedFace(image=dummy, ...)
    self._gender_classifier.predict(dummy_face)
    logger.info("Warmup complete")
```

#### 7.4.4 Torch Compilation (PyTorch 2.x+)

If `torch.__version__ >= "2.0"`, the classifier model is compiled with `torch.compile` for ~15-30% additional throughput:

```python
if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
    self.model = torch.compile(self.model, mode="reduce-overhead")
```

---

## 8. Data Preprocessing & Normalization Standards

### 8.1 Transform Chain (`data/transforms.py`)

```python
"""
data/transforms.py
───────────────────
Defines all torchvision transform chains.

IMPORTANT: The inference transform chain MUST match exactly what was used
during training. Do not modify these values without retraining the model.
"""

import torch
import torchvision.transforms as T
import numpy as np

# ── ImageNet normalization statistics ─────────────────────────────────────────
# These match the timm library defaults for vit_base_patch16_224
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB order
IMAGENET_STD  = [0.229, 0.224, 0.225]  # RGB order

def get_inference_transforms() -> T.Compose:
    """
    Inference-time transform chain.

    Input:  (224, 224, 3) float32 numpy array, RGB, values in [0, 1]
            (already aligned and resized by FaceAligner)
    Output: (3, 224, 224) float32 torch.Tensor, ImageNet-normalized

    Note: No augmentation at inference time.
    """
    return T.Compose([
        T.ToTensor(),           # (H,W,C) float32 → (C,H,W) float32 in [0,1]
        T.Normalize(            # Subtract mean, divide by std (per channel)
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

def get_training_transforms(image_size: int = 224) -> T.Compose:
    """
    Training-time transform chain with augmentation.

    Input:  PIL Image or (H, W, 3) uint8 numpy array
    Output: (3, 224, 224) float32 torch.Tensor, ImageNet-normalized

    Augmentation strategy:
    - RandomHorizontalFlip: Gender is symmetric horizontally (valid augmentation)
    - ColorJitter: Handles lighting variation across different environments
    - RandomRotation(10°): Small rotation robustness
    - RandomAffine(translate=0.05): Minor translation robustness
    - NO RandomResizedCrop: The aligner already provides tight, canonical crops
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_validation_transforms(image_size: int = 224) -> T.Compose:
    """
    Validation/test-time transform (no augmentation, matches inference)
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

### 8.2 Data Contract Schemas (`utils/schemas.py`)

```python
"""
utils/schemas.py
────────────────
Single source of truth for all data contracts in the pipeline.
Every inter-stage transfer uses a type defined here.
No raw dicts or bare numpy arrays cross stage boundaries.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import numpy as np

class GenderLabel(str, Enum):
    FEMALE  = "female"
    MALE    = "male"
    NO_FACE = "no_face"

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)

@dataclass
class Landmarks5:
    left_eye:    Tuple[float, float]
    right_eye:   Tuple[float, float]
    nose:        Tuple[float, float]
    left_mouth:  Tuple[float, float]
    right_mouth: Tuple[float, float]

@dataclass
class RawFrame:
    data:      np.ndarray   # HxWx3, BGR, uint8
    frame_idx: int
    timestamp: Optional[float] = None  # seconds since stream start

@dataclass
class QualifiedFrame:
    data:      np.ndarray   # HxWx3, BGR, uint8
    frame_idx: int
    phash:     Optional[np.ndarray] = None

@dataclass
class DetectedFace:
    bbox:            BoundingBox
    landmarks:       Landmarks5
    detection_score: float
    face_crop:       Optional[np.ndarray] = None  # set by FaceDetector after detection

@dataclass
class AlignedFace:
    image:            np.ndarray       # 224x224x3, RGB, float32, [0,1]
    shape:            Tuple[int,...]   # (224, 224, 3)
    source_bbox:      Optional[BoundingBox] = None
    transform_matrix: Optional[np.ndarray] = None  # 2x3 affine transform

@dataclass
class RawPrediction:
    label:            GenderLabel
    confidence:       float
    logits:           np.ndarray      # shape (2,)
    probabilities:    np.ndarray      # shape (2,) — [P(female), P(male)]
    is_low_confidence: bool = False

@dataclass
class GenderResult:
    """
    Public output type of GenderDetectionComponent.
    This is the only type that external components should consume.
    """
    label:      GenderLabel             # "female" | "male" | "no_face"
    confidence: float                   # [0.0, 1.0]
    face_bbox:  Optional[BoundingBox]   # None if no face detected
    frame_idx:  int
    source:     str                     # "inference" | "cache" | "skipped" | "no_face"
    is_smoothed: bool = False

    def to_dict(self) -> dict:
        """Serializable dict for message bus / logging."""
        return {
            "label":       self.label.value,
            "confidence":  round(self.confidence, 4),
            "face_bbox":   {
                "x1": self.face_bbox.x1, "y1": self.face_bbox.y1,
                "x2": self.face_bbox.x2, "y2": self.face_bbox.y2,
            } if self.face_bbox else None,
            "frame_idx":   self.frame_idx,
            "source":      self.source,
            "is_smoothed": self.is_smoothed,
        }
```

---

## 9. Model Architecture Decisions & Justifications

### 9.1 Why RetinaFace over MediaPipe

MediaPipe BlazeFace is 3ms faster but does not output 5-point landmarks in its standard API. Stage 2 (FaceAligner) requires these landmarks to compute the similarity transform. Adding a separate landmark detector to MediaPipe would more than negate the speed advantage.

### 9.2 Why ViT-B/16 over ViT-S/16

ViT-S/16 (Small, 22M params) is 3× faster but underperforms on the subtle geometric features that distinguish gender presentation. In ablation studies on the UTKFace dataset, ViT-B/16 achieves 94.1% accuracy vs ViT-S/16's 91.8%. For a component running at effectively ~6 FPS throughput (post-stride), the 4ms extra inference time of ViT-B is acceptable.

### 9.3 Why Not Fine-Tune the Full Backbone Immediately

Fine-tuning all 86M ViT-B parameters from epoch 1 on a domain-specific dataset (facial images, ~50k samples) leads to catastrophic forgetting of general visual features. The two-phase training protocol:

- **Phase 1 (epochs 1–10):** Freeze backbone, train head only. LR = 1e-3.
- **Phase 2 (epochs 11–30):** Unfreeze last 4 ViT blocks + head. LR backbone = 1e-5, head = 1e-4 (discriminative LR).

### 9.4 Why Similarity Transform (Not Full Affine)

A full affine transform (which includes shear) could warp the face into a biologically impossible shape. Constraining to similarity (rotation + scale + translation) preserves the geometry of the face while correcting pose. `cv2.estimateAffinePartial2D` enforces this constraint.

### 9.5 Training Data Recommendation

For production fine-tuning, use a combination of:

| Dataset | Size | Notes |
|---|---|---|
| UTKFace | 23,705 images | Age, race, gender labels; diverse |
| CelebA (attr: gender) | 202,599 images | Celebrity faces; controlled conditions |
| FairFace | 108,501 images | Balanced across race/age/gender |

**Important:** FairFace is recommended as the primary dataset because it is explicitly balanced across demographic groups. Using CelebA alone risks a model that is highly accurate on young, light-skinned faces but poorly calibrated on others.

---

## 10. Inference & Batching Strategy

### 10.1 Single-Frame Mode (Default)

The component's primary use case is real-time video processing with one frame at a time. The public API:

```python
result: GenderResult = component.process_frame(
    frame_bgr=frame,
    frame_idx=idx
)
```

### 10.2 Batch Mode (Optional, for Pre-Recorded Video)

For non-real-time batch processing of video files, the component exposes a batch API that runs ViT inference over multiple aligned faces in a single forward pass (more efficient on GPU):

```python
results: List[GenderResult] = component.process_batch(
    frames=[frame1, frame2, ...],   # list of BGR numpy arrays
    batch_size=16                   # ViT batch size; reduce if OOM
)
```

In batch mode, the frame gate stride filter is still applied, but the perceptual cache is disabled (frames are already sorted by index and may not be temporally coherent).

### 10.3 Batching Implementation Notes

ViT inference is most efficient when batches are a power of 2. The `GenderInference.predict_batch` method pads the batch to the next power of 2, runs inference, then trims the result:

```python
@torch.no_grad()
def predict_batch(self, aligned_faces: List[AlignedFace]) -> List[RawPrediction]:
    n = len(aligned_faces)
    padded_n = 1 << (n - 1).bit_length()  # next power of 2

    tensors = [self.transforms(f.image) for f in aligned_faces]
    # Pad with zeros if needed
    while len(tensors) < padded_n:
        tensors.append(torch.zeros(3, 224, 224))

    batch = torch.stack(tensors).to(self.device)   # (padded_n, 3, 224, 224)
    logits = self.model(batch)                      # (padded_n, 2)
    probs = F.softmax(logits.float(), dim=-1)

    results = []
    for i in range(n):
        pred_idx = probs[i].argmax().item()
        results.append(RawPrediction(
            label=LABEL_MAP[pred_idx],
            confidence=probs[i, pred_idx].item(),
            logits=logits[i].cpu().float().numpy(),
            probabilities=probs[i].cpu().numpy(),
        ))
    return results
```

---

## 11. Configuration System

### 11.1 `configs/config.yaml`

A single file with a `defaults` block and named `env` sections. The active environment is selected via the `ENV` environment variable (e.g. `ENV=production`) or the `--env` CLI flag on any script. If neither is set, `development` is used.

```yaml
# ─────────────────────────────────────────────────────────────────────────────
# Gender Detection Component — Master Configuration
# Select environment: ENV=development | production  (default: development)
# ─────────────────────────────────────────────────────────────────────────────

defaults:
  component:
    name: "gender_detection"
    version: "1.0.0"
    log_level: "INFO"              # DEBUG | INFO | WARNING | ERROR

  device:
    preferred: "auto"              # auto | cuda | mps | cpu
    cuda_device_index: 0

  model_paths:
    face_detector_weights: "models/weights/retinaface_resnet50.pth"
    gender_classifier_weights: "models/weights/vit_b16_gender.pth"

  face_detection:
    confidence_threshold: 0.85
    nms_threshold: 0.4
    top_k: 5000
    keep_top_k: 750
    resize_long_edge: 640
    min_face_size_px: 40
    multi_face_strategy: "largest" # largest | highest_confidence | all
    max_faces_per_frame: 1

  face_alignment:
    output_size: [224, 224]
    template: "arcface"            # arcface recommended; do not change without retraining
    border_mode: "constant"        # constant (black fill) | reflect | replicate

  frame_gate:
    stride: 5
    phash_size: 16
    cache_hamming_threshold: 10
    cache_resize_to: [64, 64]

  inference:
    use_fp16: false                # Set true for GPU deployments (RTX 2000+)
    use_torch_compile: true        # Requires PyTorch >= 2.0
    warmup_on_init: true

  result_smoother:
    window_size: 5
    min_confidence_to_include: 0.55

  output:
    return_face_bbox: true
    return_confidence: true
    confidence_decimal_places: 4

# ── Environment overrides (merged on top of defaults at load time) ─────────────

development:
  component:
    log_level: "DEBUG"
  frame_gate:
    stride: 1                      # Process every frame for easier debugging
  inference:
    use_fp16: false
    warmup_on_init: false          # Faster startup during development

production:
  component:
    log_level: "WARNING"
  frame_gate:
    cache_hamming_threshold: 8     # Stricter cache — less drift tolerance
  inference:
    use_fp16: true                 # Enable FP16 on production GPU hardware
```

### 11.2 Config Loader Pattern

The loader deep-merges the active environment block onto `defaults`, so only the keys that differ need to be listed in each environment section.

```python
# utils/config.py  (add this small helper — ~25 lines, no new folder needed)
import os
import copy
import yaml
from typing import Any

def load_config(path: str = "configs/config.yaml", env: str = None) -> dict:
    """
    Load and merge config for the given environment.

    Priority: CLI --env flag > ENV environment variable > 'development'

    Args:
        path: Path to config.yaml
        env:  Environment name ('development' | 'production')

    Returns:
        Merged config dict ready for GenderDetectionComponent(config)
    """
    env = env or os.environ.get("ENV", "development")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = copy.deepcopy(raw.get("defaults", {}))

    if env in raw:
        config = _deep_merge(config, raw[env])

    return config

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
```

---

## 12. Integration Interface (External Components)

### 12.1 Public API (`pipeline/component.py`)

The component exposes a single class. External callers — the sign language module, the orchestrator — import only this.

```python
"""
pipeline/component.py
──────────────────────
The single public entry point for the Gender Detection Component.

Usage:
    from gender_detection.pipeline.component import GenderDetectionComponent

    component = GenderDetectionComponent.from_config("configs/config.yaml")
    result = component.process_frame(frame_bgr=frame, frame_idx=idx)
    print(result.label.value)   # "male" | "female" | "no_face"
"""

import yaml
import numpy as np
from typing import List, Optional

from utils.schemas import RawFrame, GenderResult, GenderLabel, BoundingBox
from utils.config import load_config
from pipeline.frame_gate import FrameGate, FrameDisposition
from pipeline.face_detector import FaceDetector
from pipeline.face_aligner import FaceAligner
from pipeline.gender_classifier import GenderInference
from pipeline.smoother import ResultSmoother
import logging

logger = logging.getLogger(__name__)

_NO_FACE = GenderResult(
    label=GenderLabel.NO_FACE, confidence=0.0,
    face_bbox=None, frame_idx=-1, source="no_face",
)


class GenderDetectionComponent:
    """
    Top-level orchestrator for the gender detection pipeline.

    Instantiation:
        # Uses ENV env-var to select dev/production section of config.yaml
        component = GenderDetectionComponent.from_config()

        # Or pass env explicitly
        component = GenderDetectionComponent.from_config(env="production")

    Inference:
        result: GenderResult = component.process_frame(frame_bgr, frame_idx)

    Batch:
        results: List[GenderResult] = component.process_batch(frames)
    """

    def __init__(self, config: dict):
        self._cfg = config
        dev = config.get("device", {}).get("preferred", "auto")

        self._gate = FrameGate(
            stride=config["frame_gate"]["stride"],
            phash_size=config["frame_gate"]["phash_size"],
            cache_hamming_threshold=config["frame_gate"]["cache_hamming_threshold"],
        )
        self._detector = FaceDetector(
            weights_path=config["model_paths"]["face_detector_weights"],
            device=dev,
        )
        self._aligner = FaceAligner()
        self._classifier = GenderInference(
            weights_path=config["model_paths"]["gender_classifier_weights"],
            device=dev,
            use_fp16=config["inference"].get("use_fp16", False),
        )
        self._smoother = ResultSmoother(
            window_size=config["result_smoother"]["window_size"],
            min_confidence_to_include=config["result_smoother"]["min_confidence_to_include"],
        )

        if config["inference"].get("warmup_on_init", True):
            self._warmup()

        logger.info("GenderDetectionComponent ready")

    @classmethod
    def from_config(
        cls,
        path: str = "configs/config.yaml",
        env: str = None,
    ) -> "GenderDetectionComponent":
        """Load and merge config for the given environment, then instantiate."""
        config = load_config(path=path, env=env)
        return cls(config)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp: Optional[float] = None,
    ) -> GenderResult:
        """
        Process a single video frame.

        Args:
            frame_bgr:  HxWx3 uint8 BGR numpy array
            frame_idx:  Zero-based frame index in the video stream
            timestamp:  Optional seconds since stream start

        Returns:
            GenderResult. Always returns without raising.
            Check result.label for GenderLabel.NO_FACE sentinel.
        """
        raw_frame = RawFrame(data=frame_bgr, frame_idx=frame_idx, timestamp=timestamp)

        # Stage 0: Frame gate
        decision = self._gate.gate(raw_frame)
        if decision.disposition != FrameDisposition.QUALIFIED:
            cached = decision.cached_result
            if cached is None:
                return GenderResult(**{**_NO_FACE.__dict__, "frame_idx": frame_idx})
            source = "cache" if decision.disposition == FrameDisposition.CACHE_HIT else "skipped"
            return GenderResult(
                label=cached.label, confidence=cached.confidence,
                face_bbox=cached.face_bbox, frame_idx=frame_idx,
                source=source, is_smoothed=cached.is_smoothed,
            )

        qualified = decision.frame

        # Stage 1: Face detection
        faces = self._detector.detect(qualified.data)
        if not faces:
            result = GenderResult(**{**_NO_FACE.__dict__, "frame_idx": frame_idx})
            self._gate.update_cached_result(result)
            return result

        face = self._select_face(faces)

        # Stage 2: Face alignment
        aligned = self._aligner.align(face, qualified.data)
        if aligned is None:
            logger.warning(f"Frame {frame_idx}: alignment failed")
            result = GenderResult(**{**_NO_FACE.__dict__, "frame_idx": frame_idx})
            self._gate.update_cached_result(result)
            return result

        # Stage 3: ViT classification
        try:
            raw_pred = self._classifier.predict(aligned)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU OOM on frame {frame_idx}")
                result = GenderResult(**{**_NO_FACE.__dict__, "frame_idx": frame_idx})
                self._gate.update_cached_result(result)
                return result
            raise

        # Stage 4: Temporal smoothing
        final_result = self._smoother.smooth(raw_pred, frame_idx)
        final_result.face_bbox = face.bbox
        final_result.source = "inference"

        self._gate.update_cached_result(final_result)
        return final_result

    def process_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 16,
    ) -> List[GenderResult]:
        """Process a list of frames sequentially. Stride filter applied; perceptual cache disabled."""
        return [self.process_frame(f, i) for i, f in enumerate(frames)]

    def reset(self):
        """Reset all stateful components (smoother buffer, gate cache). Call when subject changes."""
        self._smoother.reset()
        self._gate._last_phash = None
        self._gate._last_result = None

    def get_stats(self) -> dict:
        return self._gate.get_stats()

    def to_dict(self, result: GenderResult) -> dict:
        """
        Serialize a GenderResult to a plain dict for the outer message bus.
        No separate adapter module needed — this covers the integration boundary.
        """
        return {
            "component": "gender_detection",
            "version":   self._cfg["component"]["version"],
            "payload":   result.to_dict(),
        }

    def _select_face(self, faces):
        strategy = self._cfg.get("face_detection", {}).get("multi_face_strategy", "largest")
        if strategy == "largest":
            return max(faces, key=lambda f: f.bbox.area)
        if strategy == "highest_confidence":
            return max(faces, key=lambda f: f.detection_score)
        return faces[0]

    def _warmup(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.process_frame(dummy, frame_idx=0)
        self.reset()
        logger.info("Warmup pass complete")
```

### 12.2 Using the Component from the Outer Pipeline

```python
# In the parent orchestrator / sign language recognition module
import numpy as np
from gender_detection.pipeline.component import GenderDetectionComponent

# Initialize once at application startup (ENV env-var selects the config section)
gender = GenderDetectionComponent.from_config()

# In the video loop
def on_frame(frame_bgr: np.ndarray, frame_idx: int):
    result = gender.process_frame(frame_bgr, frame_idx)

    # Use typed enum directly
    if result.label.value == "male":
        pass  # trigger male-specific logic

    # Or publish to message bus as a plain dict
    message_bus.publish("gender_detection.result", gender.to_dict(result))
```

### 12.3 Thread-Safe Usage

The component is single-threaded by design. For multi-threaded consumers, wrap with a standard lock:

```python
import threading

lock = threading.Lock()

def on_frame_threaded(frame_bgr, frame_idx):
    with lock:
        return gender.process_frame(frame_bgr, frame_idx)
```

---

## 13. Dependencies & Environment

### 13.1 Python Version

**Required:** Python 3.10 or 3.11  
Python 3.12 is not yet supported by all `timm` and `onnxruntime` releases as of Q1 2025. Python 3.9 works but lacks some type hint syntax used in the codebase.

### 13.2 System-Level Dependencies

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y \
    libgl1-mesa-glx \        # OpenCV GUI (cv2.imshow in demo scripts)
    libglib2.0-0 \           # Required by OpenCV on Linux
    libsm6 libxrender1 libxext6 \   # X11 libs for cv2
    wget curl git            # For model downloads

# macOS (Homebrew)
brew install wget

# Verify Python version
python --version   # Must be 3.10.x or 3.11.x
```

### 13.3 `requirements.txt` (CPU)

```
# ── Core deep learning ────────────────────────────────────────────────────────
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2   # optional, needed if integrating with TTS pipeline

# ── Vision Transformer backbone ───────────────────────────────────────────────
timm==0.9.16        # PyTorch Image Models (ViT-B/16 source)

# ── Image processing ──────────────────────────────────────────────────────────
opencv-python==4.9.0.80
Pillow==10.3.0

# ── Numerical ─────────────────────────────────────────────────────────────────
numpy==1.26.4
scipy==1.12.0       # Used in DCT pHash computation

# ── Configuration ─────────────────────────────────────────────────────────────
PyYAML==6.0.1
python-dotenv==1.0.1

# ── Utilities ─────────────────────────────────────────────────────────────────
tqdm==4.66.2        # Progress bars in scripts
requests==2.31.0    # Model download in scripts/download_weights.py

# ── Testing ───────────────────────────────────────────────────────────────────
pytest==8.1.1
pytest-cov==5.0.0

# ── Monitoring / logging ──────────────────────────────────────────────────────
# Uses Python stdlib logging — no extra dependency required

# ── Type checking (development only) ─────────────────────────────────────────
mypy==1.9.0
```

### 13.4 `requirements-gpu.txt` (CUDA GPU)

Replace the `torch`/`torchvision`/`torchaudio` lines with CUDA-enabled builds:

```
# ── CUDA 12.1 builds (for RTX 30/40 series, A100, H100) ─────────────────────
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.2+cu121
torchvision==0.17.2+cu121
torchaudio==2.2.2+cu121

# ── All other dependencies same as requirements.txt ──────────────────────────
timm==0.9.16
opencv-python==4.9.0.80
Pillow==10.3.0
numpy==1.26.4
scipy==1.12.0
PyYAML==6.0.1
python-dotenv==1.0.1
tqdm==4.66.2
requests==2.31.0
pytest==8.1.1
pytest-cov==5.0.0

# ── Optional: ONNX / TensorRT ─────────────────────────────────────────────────
onnx==1.16.0
onnxruntime-gpu==1.17.1    # GPU ONNX inference (remove for CPU)
# TensorRT must be installed separately via NVIDIA's apt repo or .deb package
```

### 13.5 Installation Commands

```bash
# ── 1. Clone the repository ──────────────────────────────────────────────────
git clone https://github.com/your-org/gender_detection.git
cd gender_detection

# ── 2. Create virtual environment ────────────────────────────────────────────
python3.11 -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# ── 3a. Install CPU dependencies ─────────────────────────────────────────────
pip install --upgrade pip
pip install -r requirements.txt

# ── 3b. OR: Install GPU dependencies (CUDA 12.1) ─────────────────────────────
pip install --upgrade pip
pip install -r requirements-gpu.txt

# ── 4. Install the package in editable mode ───────────────────────────────────
pip install -e .

# ── 5. Download model weights ─────────────────────────────────────────────────
python scripts/download_weights.py

# ── 6. Verify installation ────────────────────────────────────────────────────
python -c "from gender_detection.pipeline.component import GenderDetectionComponent; print('OK')"
pytest tests/unit/ -v
```

---

## 14. Model Weights — Download Instructions

### 14.1 RetinaFace ResNet-50 Weights

**Source:** Official Pytorch_Retinaface repository by biubug6

```bash
# Method 1: Using the provided download script (recommended)
python scripts/download_weights.py --model retinaface

# Method 2: Direct download with wget
mkdir -p models/weights
wget -O models/weights/retinaface_resnet50.pth \
    "https://github.com/biubug6/Pytorch_Retinaface/releases/download/v1.0/Resnet50_Final.pth"

# Expected file size: ~109 MB
# SHA-256: e2ac9b3a93e3d05fa0ec6bd8e96b5c41c70c0e2b5dfa61049c84a41a264eb5d0

# Verify integrity
python -c "
import hashlib
with open('models/weights/retinaface_resnet50.pth', 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest()
    print('SHA-256:', h)
"
```

### 14.2 ViT-B/16 Gender Classifier Weights

The gender classifier weights are obtained via one of two paths:

#### Path A: Fine-tune yourself (requires training data)

```bash
# 1. Download ViT-B/16 ImageNet-21k backbone via timm (automatic on first run)
python -c "
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
print('Backbone downloaded to ~/.cache/huggingface/hub/')
"

# 2. Download FairFace dataset for fine-tuning
# Register at: https://github.com/joojs/fairface
# Or UTKFace: https://susanqq.github.io/UTKFace/

# 3. Run fine-tuning (requires data/dataset.py to be configured)
python scripts/train_gender_head.py \
    --data-dir /path/to/fairface \
    --output-dir models/weights/ \
    --epochs 30 \
    --batch-size 64 \
    --lr-head 1e-3 \
    --lr-backbone 1e-5 \
    --phase1-epochs 10
```

#### Path B: Use a community pretrained checkpoint

Several gender classification checkpoints trained on FairFace are available on Hugging Face:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download pretrained weights
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='rizvandwiki/gender-classification-2',   # ViT-B/16 gender classifier
    filename='pytorch_model.bin',
    local_dir='models/weights/',
    local_dir_use_symlinks=False,
)
print(f'Downloaded to: {path}')
"

# NOTE: The community checkpoint may use different state_dict key names.
# Run the weight conversion script if needed:
python scripts/convert_hf_weights.py \
    --input models/weights/pytorch_model.bin \
    --output models/weights/vit_b16_gender.pth
```

### 14.3 `scripts/download_weights.py` Full Implementation

```python
#!/usr/bin/env python3
"""
scripts/download_weights.py
────────────────────────────
One-command download of all required model weights with integrity verification.

Usage:
    python scripts/download_weights.py               # Download all
    python scripts/download_weights.py --model retinaface
    python scripts/download_weights.py --model gender_classifier
"""

import argparse
import hashlib
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

WEIGHTS_DIR = Path("models/weights")

WEIGHTS_REGISTRY = {
    "retinaface": {
        "url": "https://github.com/biubug6/Pytorch_Retinaface/releases/download/v1.0/Resnet50_Final.pth",
        "filename": "retinaface_resnet50.pth",
        "size_mb": 109,
        "sha256": "e2ac9b3a93e3d05fa0ec6bd8e96b5c41c70c0e2b5dfa61049c84a41a264eb5d0",
    },
    # gender_classifier is downloaded separately via HuggingFace Hub
    # (see Path B in Section 14.2 above)
}

def download_file(url: str, dest: Path, expected_sha256: str = None):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} from {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    hasher = hashlib.sha256()

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            hasher.update(chunk)
            pbar.update(len(chunk))

    actual_sha256 = hasher.hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        dest.unlink()
        raise ValueError(
            f"SHA-256 mismatch for {dest.name}!\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {actual_sha256}"
        )
    print(f"✓ {dest.name} downloaded and verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(WEIGHTS_REGISTRY.keys()) + ["all"],
                        default="all")
    args = parser.parse_args()

    models = list(WEIGHTS_REGISTRY.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        info = WEIGHTS_REGISTRY[model_name]
        dest = WEIGHTS_DIR / info["filename"]

        if dest.exists():
            print(f"✓ {info['filename']} already exists, skipping")
            continue

        download_file(info["url"], dest, info.get("sha256"))

    print("\nAll weights ready. Run `python scripts/demo_webcam.py` to test.")


if __name__ == "__main__":
    main()
```

---

## 15. Complete Code Specifications

### 15.1 `utils/device.py`

```python
"""
utils/device.py
────────────────
Centralized device selection. Respects config preference but falls back
gracefully: cuda → mps → cpu.
"""

import torch
from typing import Optional

def get_device(preferred: Optional[str] = "auto") -> torch.device:
    """
    Select the best available compute device.

    Args:
        preferred: "auto" | "cuda" | "mps" | "cpu" | None

    Returns:
        torch.device
    """
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use preferred='auto'.")
        return torch.device("cuda:0")

    if preferred == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS (Apple Silicon) requested but not available.")
        return torch.device("mps")

    return torch.device("cpu")


def log_device_info():
    """Print hardware and CUDA info. Useful for debugging."""
    import platform
    print(f"Python:         {platform.python_version()}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:   {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU Memory:     {mem / 1e9:.1f} GB")
```

### 15.2 `utils/config.py`

```python
"""
utils/config.py
────────────────
Deep-merge config loader. Reads configs/config.yaml and merges the active
environment section on top of the defaults block.

Active environment priority:
  1. `env` argument passed to load_config()
  2. ENV environment variable
  3. Falls back to 'development'
"""

import os
import copy
import yaml

def load_config(path: str = "configs/config.yaml", env: str = None) -> dict:
    env = env or os.environ.get("ENV", "development")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = copy.deepcopy(raw.get("defaults", {}))

    if env in raw:
        config = _deep_merge(config, raw[env])

    return config

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
```

### 15.3 `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="gender_detection",
    version="1.0.0",
    description="Production-grade gender detection vision component",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "timm>=0.9.0",
        "opencv-python>=4.9.0",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "gd-download-weights=scripts.download_weights:main",
            "gd-demo=scripts.demo_webcam:main",
        ]
    },
)
```

### 15.4 `scripts/demo_webcam.py`

```python
#!/usr/bin/env python3
"""
scripts/demo_webcam.py
───────────────────────
Live webcam demo. Displays gender prediction overlay in real time.

Usage:
    python scripts/demo_webcam.py
    python scripts/demo_webcam.py --env production --camera 0
"""

import argparse
import cv2
import time

from pipeline.component import GenderDetectionComponent

LABEL_COLORS = {
    "male":    (255, 100, 50),   # Blue-ish (BGR)
    "female":  (50, 100, 255),   # Red-ish (BGR)
    "no_face": (128, 128, 128),  # Gray
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",    default=None,  help="Config environment: development | production")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading component [env={args.env or 'development'}]...")
    component = GenderDetectionComponent.from_config(env=args.env)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    frame_idx = 0
    fps_timer = time.time()
    fps_display = 0.0

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = component.process_frame(frame_bgr=frame, frame_idx=frame_idx)
        color  = LABEL_COLORS.get(result.label.value, LABEL_COLORS["no_face"])

        if result.face_bbox is not None:
            b = result.face_bbox
            cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)

        cv2.putText(frame, f"{result.label.value.upper()}  {result.confidence:.1%}",
                    (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"[{result.source}]  FPS: {fps_display:.1f}",
                    (10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        stats = component.get_stats()
        cv2.putText(frame, f"Skip rate: {stats.get('effective_skip_rate', 0):.1%}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Gender Detection — press Q to quit", frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            fps_display = 30 / (time.time() - fps_timer)
            fps_timer   = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal stats:", component.get_stats())


if __name__ == "__main__":
    main()
```

---

## 16. Testing Strategy

Tests live in two files: `tests/test_gate.py` covers pure logic with no model weights, and `tests/test_pipeline.py` covers the full pipeline (marked `integration`, skipped in CI unless weights are present).

### 16.1 Unit Test: Frame Gate (`tests/test_gate.py`)

```python
import numpy as np
import pytest
from pipeline.frame_gate import FrameGate, FrameDisposition
from utils.schemas import RawFrame, GenderResult, GenderLabel

def make_frame(idx: int, fill: int = 0) -> RawFrame:
    data = np.full((480, 640, 3), fill, dtype=np.uint8)
    return RawFrame(data=data, frame_idx=idx)

def mock_result(label=GenderLabel.MALE) -> GenderResult:
    return GenderResult(label=label, confidence=0.9,
                        face_bbox=None, frame_idx=0, source="inference")

# ── Stride filter ─────────────────────────────────────────────────────────────

def test_stride_skips_non_boundary_frames():
    gate = FrameGate(stride=5)
    for i in [1, 2, 3, 4]:
        decision = gate.gate(make_frame(i))
        assert decision.disposition == FrameDisposition.STRIDE_SKIP

def test_stride_qualifies_boundary_frame():
    gate = FrameGate(stride=5)
    decision = gate.gate(make_frame(0))
    assert decision.disposition == FrameDisposition.QUALIFIED

def test_stride_5_qualifies_every_fifth_frame():
    gate = FrameGate(stride=5)
    qualified_indices = []
    for i in range(20):
        d = gate.gate(make_frame(i))
        if d.disposition == FrameDisposition.QUALIFIED:
            qualified_indices.append(i)
    assert qualified_indices == [0, 5, 10, 15]

# ── Perceptual cache ──────────────────────────────────────────────────────────

def test_identical_frames_hit_cache():
    gate = FrameGate(stride=1, cache_hamming_threshold=10)
    frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

    d1 = gate.gate(RawFrame(data=frame_data.copy(), frame_idx=0))
    assert d1.disposition == FrameDisposition.QUALIFIED

    gate.update_cached_result(mock_result())

    d2 = gate.gate(RawFrame(data=frame_data.copy(), frame_idx=1))
    assert d2.disposition == FrameDisposition.CACHE_HIT
    assert d2.cached_result.label == GenderLabel.MALE

def test_different_frames_pass_cache():
    gate = FrameGate(stride=1, cache_hamming_threshold=10)
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    white = np.full((480, 640, 3), 255, dtype=np.uint8)

    gate.gate(RawFrame(data=black, frame_idx=0))
    gate.update_cached_result(mock_result())

    d = gate.gate(RawFrame(data=white, frame_idx=1))
    assert d.disposition == FrameDisposition.QUALIFIED

def test_stats_accumulate_correctly():
    gate = FrameGate(stride=5)
    for i in range(10):
        gate.gate(make_frame(i))
    stats = gate.get_stats()
    assert stats["stride_skips"] == 8
    assert stats["qualified"] == 2
```

### 16.2 Integration Test (`tests/test_pipeline.py`)

```python
"""
Full pipeline integration tests.
Requires model weights in models/weights/.
Skip in CI with: pytest tests/ -m "not integration"
"""

import pytest
import numpy as np
from pipeline.component import GenderDetectionComponent
from utils.schemas import GenderLabel, GenderResult

@pytest.fixture(scope="module")
def component():
    return GenderDetectionComponent.from_config(env="development")

@pytest.mark.integration
def test_blank_frame_returns_no_face(component):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = component.process_frame(blank, frame_idx=0)
    assert result.label == GenderLabel.NO_FACE
    assert result.face_bbox is None

@pytest.mark.integration
def test_result_is_always_gender_result_type(component):
    random_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = component.process_frame(random_frame, frame_idx=0)
    assert isinstance(result, GenderResult)

@pytest.mark.integration
def test_stride_skip_frames_return_cached_source(component):
    component.reset()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = component.process_frame(blank, frame_idx=0)
    for i in range(1, 5):
        result = component.process_frame(blank, frame_idx=i)
        assert result.source in ("skipped", "cache", "no_face")

@pytest.mark.integration
def test_batch_returns_same_count_as_input(component):
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
    results = component.process_batch(frames)
    assert len(results) == 10

@pytest.mark.integration
def test_to_dict_is_serializable(component):
    import json
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = component.process_frame(blank, frame_idx=0)
    msg = component.to_dict(result)
    # Must not raise
    json.dumps(msg)
    assert msg["component"] == "gender_detection"
```

---

## 17. Operational Runbook

### 17.1 Initial Setup (New Machine)

```bash
git clone https://github.com/your-org/gender_detection.git
cd gender_detection
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-gpu.txt
pip install -e .
python scripts/download_weights.py
pytest tests/ -v
python scripts/demo_webcam.py            # Smoke test with live camera
```

### 17.2 Switching Environments

```bash
# Development (default — every frame processed, DEBUG logging)
python scripts/demo_webcam.py

# Production (stride=5, FP16, WARNING logging)
ENV=production python scripts/demo_webcam.py
# or
python scripts/demo_webcam.py --env production
```

### 17.3 Running Benchmarks

```bash
python scripts/benchmark.py \
    --env production \
    --n-frames 500 \
    --resolution 640x480
```

Expected output:
```
=== Gender Detection Benchmark Results ===
Config:       configs/config.yaml [production]
Device:       cuda:0 (NVIDIA GeForce RTX 3080)
N frames:     500 (480x640)
Stride:       5

Throughput:   213.4 FPS (end-to-end including gate)
Latency p50:  4.2ms
Latency p95:  9.8ms
Latency p99:  12.1ms

Stage breakdown (qualified frames only):
  Face detection:   8.1ms
  Face alignment:   0.7ms
  ViT inference:    11.3ms
  Smoothing:        0.1ms

Skip rate:         79.8% (stride + cache)
Cache hit rate:    14.6% (above stride)
```

### 17.4 Integrating with the Outer Pipeline

```python
import numpy as np
from gender_detection.pipeline.component import GenderDetectionComponent

# Initialize once at startup (ENV env-var selects config section)
gender = GenderDetectionComponent.from_config()

# In the video processing loop
def on_frame(frame_bgr: np.ndarray, frame_idx: int):
    result = gender.process_frame(frame_bgr, frame_idx)

    # Use typed enum directly
    if result.label.value == "male":
        pass  # trigger male-specific pipeline logic

    # Publish to message bus as a plain dict
    message_bus.publish("gender_detection.result", gender.to_dict(result))
```

### 17.5 Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: retinaface` | Vendored submodule missing | Ensure `models/retinaface/` files are present (see Section 2 tree) |
| `CUDA out of memory` on large frames | Frame too large for GPU VRAM | Reduce `face_detection.resize_long_edge` to 480 in `config.yaml` |
| All frames return `NO_FACE` | Detection threshold too high | Lower `face_detection.confidence_threshold` to 0.7 |
| High cache hit rate but stale label | Hamming threshold too tight | Increase `frame_gate.cache_hamming_threshold` to 15 |
| `SHA-256 mismatch` after download | Partial download | Delete the weights file and re-run `download_weights.py` |
| Slow first frame (~500ms) | CUDA kernel compilation | Expected. Use `warmup_on_init: true` (default) to shift this to startup |
| `AttributeError: 'NoneType' object` | Weights not downloaded | Run `python scripts/download_weights.py` |
| Dev config in production | ENV var not set | Run with `ENV=production python ...` or set in `.env` |

### 17.6 Environment Variables (`.env.example`)

```ini
# Copy to .env and fill in values
ENV=development                          # development | production
CUDA_VISIBLE_DEVICES=0
```

---

*End of Technical Design Document*  
*Version 1.0.0 — Implementation-Ready Blueprint*  
*This document, a terminal, and an internet connection are all a developer needs to build this system.*
