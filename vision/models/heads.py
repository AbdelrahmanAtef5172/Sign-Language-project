import torch
import torch.nn as nn

# Index 0=Female, index 1=Male (must match trained gender_head.pt)
GENDER_LABELS = ["female", "male"]
# Index 0=Sad, index 1=Neutral, index 2=Happy (must match trained emotion_head.pt)
EMOTION_LABELS = ["sad", "neutral", "happy"]

class GenderClassificationHead(nn.Module):
    """
    Linear head for gender classification (Female, Male).
    """
    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class EmotionClassificationHead(nn.Module):
    """
    Linear head for emotion classification (Sad, Neutral, Happy).
    """
    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
