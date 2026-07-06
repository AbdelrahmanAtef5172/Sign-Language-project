import numpy as np
import cv2
import os
from typing import List, Optional

from utils.schemas import DetectedFace, BoundingBox, Landmarks5
import logging

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.5
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/opencv_dnn")
_PROTOTXT = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFEMODEL = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

class FaceDetector:
    """
    Single-responsibility wrapper around OpenCV DNN face detector.

    Uses the SSD-based Caffe model (ResNet-10 backbone) from OpenCV.
    Detected faces get estimated 5-point landmarks for alignment.
    """

    def __init__(self, weights_path: str = None, device: Optional[str] = None):
        self.model = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info(f"FaceDetector initialized (OpenCV DNN, CPU)")

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        h, w = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()

        result = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < _CONFIDENCE_THRESHOLD:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            landmarks = self._estimate_landmarks(x1, y1, x2, y2)

            result.append(DetectedFace(
                bbox=bbox,
                landmarks=landmarks,
                detection_score=float(confidence),
                face_crop=frame_bgr[y1:y2, x1:x2].copy(),
            ))

        result.sort(key=lambda f: f.detection_score, reverse=True)
        return result

    @staticmethod
    def _estimate_landmarks(x1: int, y1: int, x2: int, y2: int) -> Landmarks5:
        w = x2 - x1
        h = y2 - y1
        return Landmarks5(
            left_eye=(x1 + 0.30 * w, y1 + 0.30 * h),
            right_eye=(x1 + 0.70 * w, y1 + 0.30 * h),
            nose=(x1 + 0.50 * w, y1 + 0.52 * h),
            left_mouth=(x1 + 0.35 * w, y1 + 0.70 * h),
            right_mouth=(x1 + 0.65 * w, y1 + 0.70 * h),
        )
