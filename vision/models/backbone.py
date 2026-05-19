import torch
import torch.nn as nn
from transformers import ViTModel
from transformers import logging as hf_logging

class ViTBackbone(nn.Module):
    """
    Pretrained ViT-Small backbone for feature extraction.
    Loaded from Hugging Face transformers. Uses CLS token directly,
    so the uninitialized pooler weights are harmless — suppress the
    warning to avoid user confusion.
    """
    def __init__(self, model_name: str = "WinKawaks/vit-small-patch16-224"):
        super().__init__()
        # Pooler weights are randomly initialized (not in checkpoint) but we
        # never use outputs.pooler_output — we use last_hidden_state[:, 0, :].
        # Suppress the harmless loading warning.
        hf_logging.set_verbosity_error()
        self.vit = ViTModel.from_pretrained(model_name)
        hf_logging.set_verbosity_warning()
        
        # Freeze all backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.output_dim = self.vit.config.hidden_size # Should be 384 for vit-small
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the input tensor.
        Uses interpolate_pos_encoding to handle 96x96 inputs with 224-pretrained ViT.
        """
        outputs = self.vit(x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 0, :]
