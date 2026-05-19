import pytest
import numpy as np
import cv2
from vision.preprocessing.face_detector import FaceDetector
from vision.preprocessing.quality_filter import QualityFilter
from vision.preprocessing.transforms import VisionTransforms

def test_face_detector_no_face():
    """FaceDetector should return None if no face is in frame."""
    detector = FaceDetector()
    # Create a blank black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_crop = detector.detect_largest_face(frame)
    assert face_crop is None

def test_quality_filter_blur():
    """QualityFilter should reject blurry frames."""
    qf = QualityFilter(blur_threshold=100.0)
    # Create a perfectly smooth (blurry) gray image
    smooth_face = np.full((96, 96, 3), 128, dtype=np.uint8)
    assert qf.is_valid(smooth_face) is False

def test_quality_filter_brightness():
    """QualityFilter should reject too dark or too bright frames."""
    qf = QualityFilter(brightness_range=(20, 235))
    dark_face = np.full((96, 96, 3), 10, dtype=np.uint8)
    bright_face = np.full((96, 96, 3), 250, dtype=np.uint8)
    assert qf.is_valid(dark_face) is False
    assert qf.is_valid(bright_face) is False

def test_transforms_output_shape():
    """VisionTransforms should produce a tensor of correct shape."""
    vt = VisionTransforms(image_size=96)
    face_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tensor = vt(face_crop)
    assert tensor.shape == (1, 3, 96, 96)
