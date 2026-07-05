import cv2
import numpy as np
from typing import Tuple

class QualityFilter:
    """
    Assesses image quality to reject blurry or poorly lit frames.
    """
    def __init__(self, blur_threshold: float = 100.0, brightness_range: Tuple[int, int] = (20, 235)):
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
    
    def is_valid(self, face_crop: np.ndarray) -> bool:
        """
        Checks if the face crop meets blur and brightness requirements.
        """
        if face_crop is None or face_crop.size == 0:
            return False
            
        # 1. Check Blur (Laplacian variance)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.blur_threshold:
            return False
            
        # 2. Check Brightness (20th and 80th percentiles)
        # Using a simpler check for efficiency: mean brightness
        mean_brightness = np.mean(gray)
        if not (self.brightness_range[0] < mean_brightness < self.brightness_range[1]):
            return False
            
        return True
