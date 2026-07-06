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

import os
import numpy as np
from typing import List, Optional

from utils.schemas import RawFrame, GenderResult, GenderLabel, BoundingBox, AlignedFace
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
    """

    def __init__(self, config: dict):
        self._cfg = config
        dev = config.get("device", {}).get("preferred", "auto")

        # Adjust weight paths to be absolute or relative to the component root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        fd_weights = os.path.join(root, config["model_paths"]["face_detector_weights"])
        gc_weights = os.path.join(root, "models/weights", config["model_paths"]["gender_classifier_weights"])

        self._gate = FrameGate(
            stride=config["frame_gate"]["stride"],
            phash_size=config["frame_gate"]["phash_size"],
            cache_hamming_threshold=config["frame_gate"]["cache_hamming_threshold"],
        )
        self._detector = FaceDetector(
            weights_path=fd_weights,
            device=dev,
        )
        self._aligner = FaceAligner()
        self._classifier = GenderInference(
            weights_path=gc_weights,
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
        path: Optional[str] = None,
        env: str = None,
    ) -> "GenderDetectionComponent":
        """Load and merge config for the given environment, then instantiate."""
        if path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, "configs/config.yaml")
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
        """
        raw_frame = RawFrame(data=frame_bgr, frame_idx=frame_idx, timestamp=timestamp)

        # Stage 0: Frame gate
        decision = self._gate.gate(raw_frame)
        if decision.disposition != FrameDisposition.QUALIFIED:
            cached = decision.cached_result
            if cached is None:
                return GenderResult(label=GenderLabel.NO_FACE, confidence=0.0, face_bbox=None, frame_idx=frame_idx, source="no_face")
            source = "cache" if decision.disposition == FrameDisposition.CACHE_HIT else "skipped"
            return GenderResult(
                label=cached.label, confidence=cached.confidence,
                face_bbox=cached.face_bbox, frame_idx=frame_idx,
                source=source, is_smoothed=cached.is_smoothed,
                detection_score=cached.detection_score,
            )

        qualified = decision.frame

        # Stage 1: Face detection
        faces = self._detector.detect(qualified.data)
        if not faces:
            result = GenderResult(label=GenderLabel.NO_FACE, confidence=0.0, face_bbox=None, frame_idx=frame_idx, source="no_face")
            self._gate.update_cached_result(result)
            return result

        face = self._select_face(faces)

        # Stage 2: Face alignment
        aligned = self._aligner.align(face, qualified.data)
        if aligned is None:
            result = GenderResult(label=GenderLabel.NO_FACE, confidence=0.0, face_bbox=None, frame_idx=frame_idx, source="no_face")
            self._gate.update_cached_result(result)
            return result

        # Stage 3: ViT classification
        try:
            raw_pred = self._classifier.predict(aligned)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU OOM on frame {frame_idx}")
                result = GenderResult(label=GenderLabel.NO_FACE, confidence=0.0, face_bbox=None, frame_idx=frame_idx, source="no_face")
                self._gate.update_cached_result(result)
                return result
            raise

        # Stage 4: Temporal smoothing
        final_result = self._smoother.smooth(raw_pred, frame_idx)
        final_result.face_bbox = face.bbox
        final_result.detection_score = face.detection_score
        final_result.source = "inference"

        self._gate.update_cached_result(final_result)
        return final_result

    def process_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 16,
    ) -> List[GenderResult]:
        """Process a list of frames sequentially."""
        return [self.process_frame(f, i) for i, f in enumerate(frames)]

    def reset(self):
        """Reset all stateful components."""
        self._smoother.reset()
        self._gate._last_phash = None
        self._gate._last_result = None

    def get_stats(self) -> dict:
        return self._gate.get_stats()

    def to_dict(self, result: GenderResult) -> dict:
        """Serialize a GenderResult to a plain dict."""
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
