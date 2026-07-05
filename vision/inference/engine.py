import numpy as np
import torch
from typing import Dict, Any, Optional
from loguru import logger

from ..config import VisionConfig
from ..preprocessing.face_detector import FaceDetector
from ..preprocessing.quality_filter import QualityFilter
from ..preprocessing.transforms import VisionTransforms
from ..models.vision_model import MultiTaskVisionModel
from .temporal_smoother import PairedSmoother
from .cache import PredictionCache

class VisionInferenceEngine:
    """
    The public API for the Vision component.
    Coordinates preprocessing, model inference, and temporal smoothing.
    """
    def __init__(self, config: VisionConfig):
        self.config = config
        
        # Preprocessing
        self.face_detector = FaceDetector(config.FACE_DETECTION_CONFIDENCE)
        self.quality_filter = QualityFilter(config.QUALITY_BLUR_THRESHOLD, config.QUALITY_BRIGHTNESS_RANGE)
        self.transforms = VisionTransforms(config.IMAGE_SIZE)
        
        # Model
        self.model = MultiTaskVisionModel(config.BACKBONE_MODEL)
        
        # Inference Utilities
        self.gender_smoother = PairedSmoother(config.GENDER_SMOOTHING_WINDOW)
        self.emotion_smoother = PairedSmoother(config.EMOTION_SMOOTHING_WINDOW)
        self.cache = PredictionCache(config.CACHE_HAMMING_THRESHOLD, config.CACHE_MAX_AGE_FRAMES)
        
        # State
        self.frame_count = 0
        self.no_face_counter = 0
        self.last_result = {
            "gender": "male",
            "emotion": "neutral",
            "gender_conf": 0.0,
            "emotion_conf": 0.0,
            "data_quality": "no_data"
        }
        
    def load_model(self, gender_head_path: Optional[str] = None, emotion_head_path: Optional[str] = None):
        """Loads model weights."""
        gender_path = gender_head_path or self.config.GENDER_HEAD_PATH
        emotion_path = emotion_head_path or self.config.EMOTION_HEAD_PATH
        self.model.load_heads(gender_path, emotion_path)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single BGR video frame.
        
        Returns:
            Dictionary with gender, emotion, and confidence scores.
        """
        self.frame_count += 1
        
        # 1. Stride Check
        if self.frame_count % self.config.FRAME_STRIDE != 0:
            return self.last_result

        # 2. Face Detection
        face_crop = self.face_detector.detect_largest_face(frame)
        
        if face_crop is None:
            self.no_face_counter += 1
            if self.no_face_counter >= self.config.NO_FACE_RESET_THRESHOLD:
                self.gender_smoother.clear()
                self.emotion_smoother.clear()
                self.cache.invalidate()
            
            result = self.last_result.copy()
            result["data_quality"] = "no_face"
            return result
        
        self.no_face_counter = 0
        
        # 3. Quality Check
        if not self.quality_filter.is_valid(face_crop):
            result = self.last_result.copy()
            result["data_quality"] = "low_quality"
            return result
            
        # 4. Cache Check
        cached_prediction = self.cache.get_cached_prediction(face_crop)
        if cached_prediction:
            return self._update_smoothers_and_get_result(cached_prediction, "cache_hit")
            
        # 5. Inference
        tensor = self.transforms(face_crop)
        prediction = self.model.predict(tensor)
        
        # 6. Update Cache
        self.cache.update(face_crop, prediction)
        
        # 7. Smoothing and Final Result
        return self._update_smoothers_and_get_result(prediction, "inference_run")

    def _update_smoothers_and_get_result(self, prediction: Dict[str, Any], quality_tag: str) -> Dict[str, Any]:
        """
        Updates temporal smoothers with new raw prediction and returns stable result.
        Low-confidence predictions are rejected to prevent flickering.
        """
        if prediction["gender_conf"] >= self.config.GENDER_CONFIDENCE_THRESHOLD:
            self.gender_smoother.add(prediction["gender"], prediction["gender_conf"])
        if prediction["emotion_conf"] >= self.config.EMOTION_CONFIDENCE_THRESHOLD:
            self.emotion_smoother.add(prediction["emotion"], prediction["emotion_conf"])
        
        stable_gender, gender_conf = self.gender_smoother.get()
        stable_emotion, emotion_conf = self.emotion_smoother.get()
        
        self.last_result = {
            "gender": stable_gender or "male",
            "emotion": stable_emotion or "neutral",
            "gender_conf": gender_conf,
            "emotion_conf": emotion_conf,
            "data_quality": quality_tag
        }
        return self.last_result

    async def process_frame_async(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Async wrapper for process_frame to be used in FastAPI/WebSockets.
        Runs inference in a thread pool.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        # In a real app, we'd use a persistent executor. 
        # For this component implementation, we assume run_in_executor(None, ...) is fine.
        return await loop.run_in_executor(None, self.process_frame, frame)
