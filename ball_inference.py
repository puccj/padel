"""
YOLO inference script for ball detection on raw videos.
Processes all videos in data/raw/ and saves results to data/intermediate/ as CSV files.
"""

import os
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def process_video(video_path: str, model_path: str, output_dir: str, force_reprocess: bool = False) -> str:
    """
    Process a single video with YOLO model and save ball detections to CSV.
    
    Parameters
    ----------
    video_path : str
        Path to the input video file
    model_path : str
        Path to the YOLO model weights
    output_dir : str
        Directory to save the output CSV file
    force_reprocess : bool
        If True, reprocess even if output file already exists
        
    Returns
    -------
    str
        Path to the output CSV file
    """
    video_name = Path(video_path).stem
    output_csv = os.path.join(output_dir, f"{video_name}_balls.csv")
    
    # Check if output already exists
    if os.path.exists(output_csv) and not force_reprocess:
        print(f"Output file {output_csv} already exists. Skipping {video_name}.")
        return output_csv
    
    print(f"Processing {video_name}...")
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames at {fps:.2f} FPS")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_num', 'ball_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
        
        frame_num = 0
        ball_id_counter = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO inference
            results = model(frame, verbose=False, conf=0.1)
            
            # Extract ball detections (class ID 0)
            ball_detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Check if it's a ball (class ID 0)
                        if int(box.cls.item()) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf.item()
                            
                            ball_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence
                            })
            
            # Write detections to CSV
            if ball_detections:
                for detection in ball_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    writer.writerow([frame_num, ball_id_counter, x1, y1, x2, y2, confidence])
                    ball_id_counter += 1
            else:
                # Write empty row if no balls detected
                writer.writerow([frame_num, None, None, None, None, None, None])
            
            frame_num += 1
            
            # Progress indicator
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    cap.release()
    print(f"Completed {video_name}. Saved {frame_num} frames to {output_csv}")
    
    return output_csv


def main():
    """Main function to process all videos in data/raw/ directory."""
    parser = argparse.ArgumentParser(description='Run YOLO inference on raw videos for ball detection')
    parser.add_argument('--model', type=str, default='weights/two_classes_best.pt',
                       help='Path to YOLO model weights (default: weights/two_classes_best.pt)')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory containing videos (default: data/raw)')
    parser.add_argument('--output-dir', type=str, default='data/intermediate',
                       help='Output directory for CSV files (default: data/intermediate)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of videos even if output files exist')
    parser.add_argument('--video', type=str, default=None,
                       help='Process only a specific video file (optional)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Get list of videos to process
    if args.video:
        # Process single video
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        video_files = [args.video]
    else:
        # Process all videos in input directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(args.input_dir).glob(f'*{ext}'))
        video_files = [str(f) for f in video_files]
    
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    print(f"Model: {args.model}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    # Process each video
    processed_files = []
    for video_path in video_files:
        try:
            output_csv = process_video(video_path, args.model, args.output_dir, args.force)
            processed_files.append(output_csv)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    print("-" * 50)
    print(f"Processing complete! Generated {len(processed_files)} CSV files:")
    for csv_file in processed_files:
        print(f"  - {csv_file}")


if __name__ == "__main__":
    main()
