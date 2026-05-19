from dataclasses import dataclass
from typing import Tuple

@dataclass
class VisionConfig:
    """Configuration constants for the Vision component."""
    
    # Face Detection
    FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_MIN_SIZE: int = 64
    
    # Quality Filtering
    QUALITY_BLUR_THRESHOLD: float = 100.0
    QUALITY_BRIGHTNESS_RANGE: Tuple[int, int] = (20, 235)
    
    # Model Architecture
    BACKBONE_MODEL: str = "WinKawaks/vit-small-patch16-224"
    IMAGE_SIZE: int = 224
    
    # Temporal Smoothing
    GENDER_SMOOTHING_WINDOW: int = 30
    EMOTION_SMOOTHING_WINDOW: int = 15
    
    # Confidence Thresholds
    GENDER_CONFIDENCE_THRESHOLD: float = 0.7
    EMOTION_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Prediction Cache
    CACHE_HAMMING_THRESHOLD: int = 5
    CACHE_MAX_AGE_FRAMES: int = 60
    CACHE_MAX_SIZE: int = 100
    
    # Inference Strategy
    FRAME_STRIDE: int = 3
    NO_FACE_RESET_THRESHOLD: int = 90
    
    # Paths
    GENDER_HEAD_PATH: str = "weights/vision/gender_head.pt"
    EMOTION_HEAD_PATH: str = "weights/vision/emotion_head.pt"
