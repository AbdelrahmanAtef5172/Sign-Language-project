import time
import numpy as np
import torch
import sys
import os
from loguru import logger

# Add project root to sys.path
sys.path.append(os.getcwd())

from vision.config import VisionConfig
from vision.inference.engine import VisionInferenceEngine

def benchmark_vision():
    """
    Measures Vision component latency using dummy frames.
    """
    config = VisionConfig(FRAME_STRIDE=1)
    engine = VisionInferenceEngine(config)
    
    # Pre-heat model
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.info("Pre-heating model...")
    for _ in range(5):
        engine.process_frame(dummy_frame)
        
    num_frames = 100
    latencies = []
    
    logger.info(f"Running benchmark with {num_frames} frames...")
    
    for i in range(num_frames):
        # Create a "face-like" blob to pass face detection if needed, 
        # or just use random noise and accept "no_face" latency if testing full pipeline.
        # Here we test the full process_frame logic.
        start_time = time.perf_counter()
        engine.process_frame(dummy_frame)
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n--- Vision Benchmark Results ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"95th Percentile: {p95_latency:.2f} ms")
    print(f"Target Latency:  50.00 ms (CPU)")
    
    if avg_latency < 50:
        print("Status: PASSED")
    else:
        print("Status: FAILED (Exceeds target)")

if __name__ == "__main__":
    benchmark_vision()
