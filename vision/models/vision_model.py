import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .backbone import ViTBackbone
from .heads import GenderClassificationHead, EmotionClassificationHead

class MultiTaskVisionModel(nn.Module):
    """
    Combines a shared ViT backbone with task-specific heads for 
    gender and emotion classification.
    """
    def __init__(self, model_name: str = "WinKawaks/vit-small-patch16-224"):
        super().__init__()
        self.backbone = ViTBackbone(model_name)
        self.gender_head = GenderClassificationHead(self.backbone.output_dim)
        self.emotion_head = EmotionClassificationHead(self.backbone.output_dim)
        
        # Label Mappings as per Step 2 requirements:
        # gender: index 0=female, index 1=male
        # emotion: index 0=sad, index 1=neutral, index 2=happy
        self.gender_labels = ["female", "male"]
        self.emotion_labels = ["sad", "neutral", "happy"]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone and all heads.
        """
        features = self.backbone(x)
        gender_logits = self.gender_head(features)
        emotion_logits = self.emotion_head(features)
        
        return {
            "gender_logits": gender_logits,
            "emotion_logits": emotion_logits
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Performs inference and returns human-readable labels and confidences.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            gender_probs = F.softmax(outputs["gender_logits"], dim=1)
            emotion_probs = F.softmax(outputs["emotion_logits"], dim=1)
            
            gender_conf, gender_idx = torch.max(gender_probs, dim=1)
            emotion_conf, emotion_idx = torch.max(emotion_probs, dim=1)
            
            return {
                "gender": self.gender_labels[gender_idx.item()],
                "emotion": self.emotion_labels[emotion_idx.item()],
                "gender_conf": gender_conf.item(),
                "emotion_conf": emotion_conf.item()
            }

    def load_heads(self, gender_path: str, emotion_path: str):
        """Loads pretrained weights for the classification heads."""
        import os
        if not os.path.exists(gender_path):
            raise RuntimeError(f"Missing gender head weights at {gender_path}")
        if not os.path.exists(emotion_path):
            raise RuntimeError(f"Missing emotion head weights at {emotion_path}")
            
        # Map keys from SVD script (weight/bias) to Head classes (fc.weight/fc.bias)
        g_sd = torch.load(gender_path, map_location="cpu")
        e_sd = torch.load(emotion_path, map_location="cpu")
        
        g_sd_mapped = {f"fc.{k}": v for k, v in g_sd.items() if k in ["weight", "bias"]}
        e_sd_mapped = {f"fc.{k}": v for k, v in e_sd.items() if k in ["weight", "bias"]}
        
        self.gender_head.load_state_dict(g_sd_mapped)
        self.emotion_head.load_state_dict(e_sd_mapped)
        
        from loguru import logger
        logger.info(f"Loaded vision heads from {gender_path} and {emotion_path}")
