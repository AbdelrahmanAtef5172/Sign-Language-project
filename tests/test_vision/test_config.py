import pytest
from vision.config import VisionConfig

def test_vision_config_instantiation():
    """Verify VisionConfig can be instantiated with defaults."""
    config = VisionConfig()
    assert config.FACE_DETECTION_CONFIDENCE == 0.5
    assert config.IMAGE_SIZE == 96
    assert config.FRAME_STRIDE == 3

def test_vision_config_custom_values():
    """Verify VisionConfig accepts custom values."""
    config = VisionConfig(FRAME_STRIDE=1, IMAGE_SIZE=224)
    assert config.FRAME_STRIDE == 1
    assert config.IMAGE_SIZE == 224
