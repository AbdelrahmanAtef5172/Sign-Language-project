import cv2
import time
import sys
import os
from loguru import logger

# Add project root to sys.path
sys.path.append(os.getcwd())

from vision.config import VisionConfig
from vision.inference.engine import VisionInferenceEngine

def run_demo():
    """
    Runs a standalone Vision demo using the default webcam.
    Displays gender and emotion predictions in real-time.
    """
    config = VisionConfig(FRAME_STRIDE=1) # Process every frame for demo smoothness
    engine = VisionInferenceEngine(config)
    
    # Load models if weights exist
    try:
        engine.load_model()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load pretrained weights: {e}. Using uninitialized model.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    logger.info("Starting webcam demo. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        result = engine.process_frame(frame)
        latency = (time.time() - start_time) * 1000
        
        # Display results on frame
        gender = result["gender"]
        emotion = result["emotion"]
        g_conf = result["gender_conf"]
        e_conf = result["emotion_conf"]
        quality = result["data_quality"]
        
        text = f"{gender} ({g_conf:.2f}), {emotion} ({e_conf:.2f}) | {quality} | {latency:.1f}ms"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Vision Component Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
