import pytest
import numpy as np
from vision.config import VisionConfig
from vision.inference.engine import VisionInferenceEngine

def test_engine_stride(mocker):
    """Verify engine respect frame stride."""
    # Mock dependencies to avoid real processing
    # Return a dummy face crop to avoid cv2 errors in quality filter
    dummy_face = np.zeros((96, 96, 3), dtype=np.uint8)
    mocker.patch("vision.preprocessing.face_detector.FaceDetector.detect_largest_face", return_value=dummy_face)
    mocker.patch("vision.preprocessing.quality_filter.QualityFilter.is_valid", return_value=False) # Force quality fail for simplicity
    mocker.patch("vision.models.backbone.ViTModel.from_pretrained")
    
    config = VisionConfig(FRAME_STRIDE=3)
    engine = VisionInferenceEngine(config)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Frame 1: Stride skip
    res1 = engine.process_frame(frame)
    assert res1["data_quality"] == "no_data"
    
    # Frame 2: Stride skip
    res2 = engine.process_frame(frame)
    assert res2["data_quality"] == "no_data"
    
    # Frame 3: Process (will be low_quality because we mocked is_valid to return False)
    res3 = engine.process_frame(frame)
    assert res3["data_quality"] == "low_quality"

def test_engine_no_face_reset(mocker):
    """Verify smoothers reset after many no-face frames."""
    mocker.patch("vision.preprocessing.face_detector.FaceDetector.detect_largest_face", return_value=None)
    mocker.patch("vision.models.backbone.ViTModel.from_pretrained")
    
    config = VisionConfig(FRAME_STRIDE=1, NO_FACE_RESET_THRESHOLD=5)
    engine = VisionInferenceEngine(config)
    
    # Fill smoother history (mocking internal call)
    engine.gender_smoother.add("female", 0.9)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        engine.process_frame(frame)
        
    assert engine.gender_smoother.label_smoother.get_stable_prediction() is None
