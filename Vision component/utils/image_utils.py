"""
utils/image_utils.py
────────────────────
Helpers for BGR/RGB conversion, resizing, padding, and cropping.
"""

import cv2
import numpy as np
from typing import Tuple

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image (OpenCV default) to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def resize_and_pad(
    image: np.ndarray,
    target_size: Tuple[int, int],
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image to fit within target_size while maintaining aspect ratio,
    padding the rest with border_value.
    """
    h, w = image.shape[:2]
    th, tw = target_size
    
    scale = min(tw / w, th / h)
    nh, nw = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=border_mode,
        value=border_value
    )
    return padded

def crop_box(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Surgically crop a region from an image with boundary checking."""
    h, w = image.shape[:2]
    ix1, iy1, ix2, iy2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    return image[iy1:iy2, ix1:ix2].copy()
