"""
Utilities for the gender detection vision component.
"""

from .schemas import (
    GenderLabel,
    BoundingBox,
    Landmarks5,
    RawFrame,
    QualifiedFrame,
    DetectedFace,
    AlignedFace,
    RawPrediction,
    GenderResult
)
from .device import get_device
from .transforms import get_inference_transforms, get_training_transforms
