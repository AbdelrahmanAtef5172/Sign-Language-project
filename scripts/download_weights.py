import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loguru import logger

def generate_dummy_weights(target_path, input_dim, output_dim):
    """Generates a dummy .pt file for a linear layer."""
    linear = nn.Linear(input_dim, output_dim)
    torch.save(linear.state_dict(), target_path)
    logger.info(f"Generated dummy weights at {target_path}")

def download_vision_weights():
    """
    Downloads pretrained weights for the Vision component heads 
    from the Hugging Face repository. Falls back to dummy weights if download fails.
    """
    repo_id = "WinKawaks/vit-small-patch16-224"
    target_dir = "weights/vision"
    
    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # task -> (filename, output_dim)
    tasks = {
        "gender": ("gender_head.pt", 2),
        "emotion": ("emotion_head.pt", 3)
    }
    
    input_dim = 384 # ViT-Small hidden dim
    
    logger.info(f"Attempting to download Vision weights from {repo_id}...")
    
    for task, (filename, output_dim) in tasks.items():
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            logger.info(f"File {filename} already exists. Skipping.")
            continue
            
        try:
            logger.info(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"Successfully downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            logger.warning(f"Generating dummy weights for {task} to allow testing...")
            generate_dummy_weights(target_path, input_dim, output_dim)

if __name__ == "__main__":
    download_vision_weights()
    logger.info("Weight preparation completed.")
