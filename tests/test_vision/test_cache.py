import pytest
import numpy as np
from vision.inference.cache import PredictionCache

def test_cache_hit_identical():
    """Identical frames should result in a cache hit."""
    cache = PredictionCache(hamming_threshold=0)
    face_crop = np.zeros((96, 96, 3), dtype=np.uint8)
    prediction = {"gender": "male", "emotion": "neutral"}
    
    cache.update(face_crop, prediction)
    cached = cache.get_cached_prediction(face_crop)
    
    assert cached == prediction

def test_cache_miss_different():
    """Different frames should result in a cache miss."""
    cache = PredictionCache(hamming_threshold=0)
    face1 = np.zeros((96, 96, 3), dtype=np.uint8)
    face2 = np.ones((96, 96, 3), dtype=np.uint8) * 255
    prediction = {"gender": "male", "emotion": "neutral"}
    
    cache.update(face1, prediction)
    cached = cache.get_cached_prediction(face2)
    
    assert cached is None

def test_cache_expiration():
    """Cache should expire after max_age_frames."""
    cache = PredictionCache(max_age_frames=2)
    face_crop = np.zeros((96, 96, 3), dtype=np.uint8)
    prediction = {"gender": "male", "emotion": "neutral"}
    
    cache.update(face_crop, prediction)
    # Use the same frame twice
    cache.get_cached_prediction(face_crop) # frames_since_last = 1
    cache.get_cached_prediction(face_crop) # frames_since_last = 2
    cached = cache.get_cached_prediction(face_crop) # frames_since_last = 3 -> Expired
    
    assert cached is None
