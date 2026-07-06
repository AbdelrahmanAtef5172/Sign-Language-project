"""
pipeline/smoother.py
────────────────────────────
Temporal majority-vote smoother. Reduces single-frame prediction noise
by averaging over a sliding window of recent predictions.

Input:  RawPrediction
Output: GenderResult (smoothed)
"""

from collections import deque
from typing import Deque
import numpy as np

from utils.schemas import RawPrediction, GenderResult, GenderLabel

class ResultSmoother:
    """
    Sliding window majority-vote smoother.
    """

    def __init__(self, window_size: int = 5, min_confidence_to_include: float = 0.55):
        self.window_size = window_size
        self.min_confidence = min_confidence_to_include
        self._buffer: Deque[RawPrediction] = deque(maxlen=window_size)

    def smooth(self, raw: RawPrediction, frame_idx: int) -> GenderResult:
        """
        Add raw prediction to buffer and return smoothed result.
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
        label_idx = 1 if voted_label == GenderLabel.FEMALE else 0
        avg_confidence = float(np.mean([p.probabilities[label_idx] for p in self._buffer]))

        return GenderResult(
            label=voted_label,
            confidence=avg_confidence,
            face_bbox=raw.aligned_face.source_bbox if hasattr(raw, 'aligned_face') and raw.aligned_face else None,
            frame_idx=frame_idx,
            source="inference_smoothed",
            is_smoothed=True,
        )

    def reset(self):
        """Clear the smoothing buffer. Call when subject changes."""
        self._buffer.clear()
