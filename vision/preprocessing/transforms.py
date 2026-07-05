import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class VisionTransforms:
    """
    Applies inference-only transforms to prepare face crops for the ViT model.
    """
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    def __call__(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Transforms BGR NumPy array to normalized PyTorch tensor.
        """
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) if face_crop.shape[2] == 3 else face_crop
        return self.transform(rgb_crop).unsqueeze(0)
