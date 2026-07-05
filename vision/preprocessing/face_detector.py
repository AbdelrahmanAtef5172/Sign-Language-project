import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

class FaceDetector:
    """
    Uses MediaPipe Face Detection to extract the largest face from a frame.
    Performs alignment and cropping.
    """
    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1, # 1 for full-range model (better for distant faces)
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_largest_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces, picks the largest one, and returns a cropped BGR image.
        
        Returns:
            Crop of the face as np.ndarray, or None if no face detected.
        """
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return None
            
        # Find largest face by bounding box area
        largest_detection = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width * 
                         d.location_data.relative_bounding_box.height
        )
        
        # Extract bounding box
        h, w, _ = frame.shape
        bbox = largest_detection.location_data.relative_bounding_box
        
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Add some padding (20%)
        padx = int(width * 0.2)
        pady = int(height * 0.2)
        
        xmin = max(0, xmin - padx)
        ymin = max(0, ymin - pady)
        xmax = min(w, xmin + width + 2*padx)
        ymax = min(h, ymin + height + 2*pady)
        
        face_crop = frame[ymin:ymax, xmin:xmax]
        
        if face_crop.size == 0:
            return None
            
        return face_crop

    def align_face(self, face_crop: np.ndarray, target_size: int = 96) -> np.ndarray:
        """
        Resizes face crop to target size. 
        In a full implementation, this would also apply affine transforms for alignment.
        """
        return cv2.resize(face_crop, (target_size, target_size))
