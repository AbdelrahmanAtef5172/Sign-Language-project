"""
scripts/process_video.py
────────────────────────
Offline batch processor for video files. 
Extracts gender predictions for every frame and saves them to a JSON file.
"""

import cv2
import os
import sys
import json
import time
from tqdm import tqdm
from argparse import ArgumentParser

# Add root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

from pipeline.component import GenderDetectionComponent

def main():
    parser = ArgumentParser(description="Process a video file for gender detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to output JSON file (default: input_base.json)")
    parser.add_argument("--config", type=str, help="Path to config yaml")
    parser.add_argument("--env", type=str, default="production", help="Environment (development|production)")
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + "_gender.json"

    print(f"Initializing Gender Detection Component (env={args.env})...")
    component = GenderDetectionComponent.from_config(path=args.config, env=args.env)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {total_frames} frames at {fps:.2f} FPS...")

    results = []
    start_time = time.time()

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        result = component.process_frame(frame, frame_idx, timestamp=timestamp)
        
        # Only store non-trivial results to save space, or store everything?
        # Let's store everything but in a compact dict
        results.append(result.to_dict())

    total_time = time.time() - start_time
    print(f"\nProcessing complete in {total_time:.2f}s ({total_frames/total_time:.2f} FPS)")

    # Save results
    output_data = {
        "metadata": {
            "input_file": args.input,
            "total_frames": total_frames,
            "fps": fps,
            "processing_time_sec": total_time,
            "component_version": "1.0.0"
        },
        "frames": results
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")

    # Print summary stats
    stats = component.get_stats()
    print("\nPipeline Statistics:")
    print(f"  Qualified for inference: {stats['qualified']}")
    print(f"  Cache hits (pHash):      {stats['cache_hits']}")
    print(f"  Stride skips:            {stats['stride_skips']}")

if __name__ == "__main__":
    main()
