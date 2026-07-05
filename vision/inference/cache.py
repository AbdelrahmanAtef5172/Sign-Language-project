import imagehash
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time

@dataclass
class CacheEntry:
    prediction: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    frame_count: int = 0

class PredictionCache:
    """
    Caches model predictions based on perceptual hashing of face crops.
    Avoids recomputing inference for nearly identical frames.
    """
    def __init__(self, hamming_threshold: int = 5, max_age_frames: int = 60):
        self.hamming_threshold = hamming_threshold
        self.max_age_frames = max_age_frames
        self.last_hash: Optional[imagehash.ImageHash] = None
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.frames_since_last_inference: int = 0
    
    def get_cached_prediction(self, face_crop: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Computes hash of current crop and returns cached prediction if distance is low.
        """
        if self.last_prediction is None:
            return None
            
        if self.frames_since_last_inference >= self.max_age_frames:
            return None
            
        current_hash = imagehash.phash(Image.fromarray(face_crop))
        
        if self.last_hash is not None:
            distance = current_hash - self.last_hash
            if distance <= self.hamming_threshold:
                self.frames_since_last_inference += 1
                return self.last_prediction
        
        return None
    
    def update(self, face_crop: np.ndarray, prediction: Dict[str, Any]):
        """Updates the cache with a new inference result."""
        self.last_hash = imagehash.phash(Image.fromarray(face_crop))
        self.last_prediction = prediction
        self.frames_since_last_inference = 0
    
    def invalidate(self):
        """Clears the cache."""
        self.last_hash = None
        self.last_prediction = None
        self.frames_since_last_inference = 0
