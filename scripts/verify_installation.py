import sys
import os
from loguru import logger

# Add project root to sys.path
sys.path.append(os.getcwd())

def verify_imports():
    """
    Attempts to import all major dependencies and reporting any failures.
    """
    dependencies = [
        "fastapi", "uvicorn", "websockets", "torch", "torchvision", 
        "transformers", "onnxruntime", "cv2", "mediapipe", "PIL", 
        "imagehash", "numpy", "scipy", "pydantic", "dotenv", "loguru"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == "cv2":
                import cv2
            elif dep == "PIL":
                from PIL import Image
            elif dep == "dotenv":
                import dotenv
            else:
                __import__(dep)
            logger.info(f"Successfully imported {dep}")
        except ImportError as e:
            logger.error(f"Failed to import {dep}: {e}")
            missing.append(dep)
            
    if missing:
        logger.critical(f"Missing dependencies: {', '.join(missing)}")
        sys.exit(1)
    else:
        logger.info("All major dependencies are correctly installed.")

def verify_project_imports():
    """
    Verifies that project packages can be imported.
    """
    try:
        from shared.session_state import SessionState
        from vision.config import VisionConfig
        from vision.inference.engine import VisionInferenceEngine
        logger.info("Project package imports successful.")
    except ImportError as e:
        logger.error(f"Failed to import project packages: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_imports()
    verify_project_imports()
    print("\nInstallation verification PASSED.")
