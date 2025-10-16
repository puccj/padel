"""
Visualization script for raw YOLO detections (before filtering).
This helps us see what the model is actually detecting before any postprocessing.
"""

import os
import csv
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_raw_detections_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load raw ball detections from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame_num, ball_id, x1, y1, x2, y2, confidence
    """
    df = pd.read_csv(csv_path)
    
    # Filter out rows with no detections (where ball_id is NaN)
    df = df.dropna(subset=['ball_id'])
    
    # Convert ball_id to int
    df['ball_id'] = df['ball_id'].astype(int)
    
    return df


def draw_ball_detection(frame: np.ndarray, 
                       bbox: List[float], 
                       ball_id: int, 
                       confidence: float,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw a ball detection on the frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input frame
    bbox : List[float]
        Bounding box [x1, y1, x2, y2]
    ball_id : int
        Ball track ID
    confidence : float
        Detection confidence
    color : Tuple[int, int, int]
        BGR color for the bounding box
        
    Returns
    -------
    np.ndarray
        Frame with ball detection drawn
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw ball ID and confidence
    label = f"Ball {ball_id} ({confidence:.2f})"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    
    # Draw label background
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                  (x1 + label_size[0], y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def create_raw_visualization_video(input_video: str, 
                                  detections_csv: str, 
                                  output_video: str,
                                  fps_multiplier: float = 1.0) -> None:
    """
    Create a visualization video with raw ball detections overlaid.
    
    Parameters
    ----------
    input_video : str
        Path to the input video file
    detections_csv : str
        Path to the CSV file with raw ball detections
    output_video : str
        Path to the output visualization video
    fps_multiplier : float
        Speed multiplier for output video (1.0 = normal speed)
    """
    print(f"Creating raw visualization: {input_video} + {detections_csv} -> {output_video}")
    
    # Load detections
    df = load_raw_detections_from_csv(detections_csv)
    if df.empty:
        print("No detections found in CSV file.")
        return
    
    # Group detections by frame
    frame_detections = df.groupby('frame_num')
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps * fps_multiplier
    out = cv2.VideoWriter(output_video, fourcc, out_fps, (width, height))
    
    # Colors for different balls (cycle through colors)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    frame_num = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_detection_count = 0
        
        # Get detections for this frame
        if frame_num in frame_detections.groups:
            frame_detections_group = frame_detections.get_group(frame_num)
            
            for _, detection in frame_detections_group.iterrows():
                ball_id = int(detection['ball_id'])  # Convert to int
                bbox = [detection['x1'], detection['y1'], detection['x2'], detection['y2']]
                confidence = detection['confidence']
                
                # Choose color for this ball
                color = colors[ball_id % len(colors)]
                
                # Draw detection
                frame = draw_ball_detection(frame, bbox, ball_id, confidence, color)
                frame_detection_count += 1
                total_detections += 1
        
        # Add frame number and statistics
        stats_text = f"Frame: {frame_num}/{total_frames} | Detections: {frame_detection_count} | Total: {total_detections}"
        cv2.putText(frame, stats_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add title
        cv2.putText(frame, "RAW YOLO DETECTIONS (No Filtering)", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        frame_num += 1
        
        # Progress indicator
        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Raw visualization complete! Saved to {output_video}")
    print(f"Total detections processed: {total_detections}")


def main():
    """Main function for raw ball detection visualization."""
    parser = argparse.ArgumentParser(description='Create visualization videos with raw YOLO ball detections')
    parser.add_argument('--input-video', type=str, required=True,
                       help='Path to input video file')
    # --detections-csv argument is now optional and not required
    parser.add_argument('--detections-csv', type=str, default=None,
                       help='Path to CSV file with raw ball detections (default: use name from input video)')
    parser.add_argument('--output-video', type=str, default=None,
                       help='Path to output visualization video (default: auto-generated)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speed multiplier for output video (default: 1.0)')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Directory containing input videos (for batch processing)')
    parser.add_argument('--detections-dir', type=str, default='data/intermediate',
                       help='Directory containing raw detection CSV files (for batch processing and single file default)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory for output visualization videos (default: data/processed)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all videos in input directory')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        print("Batch processing mode for raw detections...")
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(args.input_dir).glob(f'*{ext}'))
        video_files = [str(f) for f in video_files]
        
        if not video_files:
            print(f"No video files found in {args.input_dir}")
            return
        
        print(f"Found {len(video_files)} video(s) to process")
        
        # Process each video
        for video_path in video_files:
            video_name = Path(video_path).stem
            
            # Find corresponding raw CSV file using the raw video name
            csv_file = os.path.join(args.detections_dir, f"{video_name}_balls.csv")
            if not os.path.exists(csv_file):
                print(f"No raw detection file found for {video_name}: {csv_file}")
                continue
            
            # Generate output filename
            output_video = os.path.join(args.output_dir, f"{video_name}_raw_visualization.mp4")
            
            try:
                create_raw_visualization_video(
                    video_path,
                    csv_file,
                    output_video,
                    fps_multiplier=args.speed
                )
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                continue
    
    else:
        # Single file processing mode
        if not os.path.exists(args.input_video):
            raise FileNotFoundError(f"Input video not found: {args.input_video}")
        
        video_name = Path(args.input_video).stem

        # If --detections-csv is not provided, use naming convention from raw video name and detections-dir
        if args.detections_csv is None:
            detections_csv = os.path.join(args.detections_dir, f"{video_name}_balls.csv")
        else:
            detections_csv = args.detections_csv

        if not os.path.exists(detections_csv):
            raise FileNotFoundError(f"Detections CSV not found: {detections_csv}")
        
        # Generate output filename if not provided
        if args.output_video is None:
            output_video = os.path.join(args.output_dir, f"{video_name}_raw_visualization.mp4")
        else:
            output_video = args.output_video
        
        create_raw_visualization_video(
            args.input_video,
            detections_csv,
            output_video,
            fps_multiplier=args.speed
        )


if __name__ == "__main__":
    main()
