"""
Ball postprocessing script to assign consistent IDs to ball detections.
Loads CSV files from data/intermediate/ and applies simple tracking.
"""

import os
import csv
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_detections_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load ball detections from CSV file.
    
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


class SimpleBallTracker:
    """Simple ball tracker that maintains consistent IDs across frames."""
    
    def __init__(self, max_distance: float = 150.0, max_lost_frames: int = 120):
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.tracks = {}  # {track_id: {'centroid': (x, y), 'last_seen': frame_num}}
        self.next_track_id = 1
        
    def calculate_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update(self, detections: List[Dict], frame_num: int) -> List[Dict]:
        # Calculate centroids for all detections
        detection_centroids = []
        for det in detections:
            centroid = self.calculate_centroid(det['bbox'])
            detection_centroids.append((centroid, det))
        
        # Associate detections with existing tracks
        used_detections = set()
        updated_detections = []
        
        # Try to associate each existing track with the closest detection
        for track_id, track_info in self.tracks.items():
            best_distance = float('inf')
            best_detection_idx = None
            
            for i, (centroid, det) in enumerate(detection_centroids):
                if i in used_detections:
                    continue
                    
                distance = self.calculate_distance(track_info['centroid'], centroid)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            if best_detection_idx is not None:
                # Update existing track
                centroid, det = detection_centroids[best_detection_idx]
                track_info['centroid'] = centroid
                track_info['last_seen'] = frame_num
                
                det['track_id'] = track_id
                updated_detections.append(det)
                used_detections.add(best_detection_idx)
        
        # Create new tracks for unassociated detections
        for i, (centroid, det) in enumerate(detection_centroids):
            if i not in used_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[track_id] = {
                    'centroid': centroid,
                    'last_seen': frame_num
                }
                
                det['track_id'] = track_id
                updated_detections.append(det)
        
        # Remove old tracks (not seen for too long)
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if frame_num - track_info['last_seen'] > self.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return updated_detections


def filter_tracks(tracked_detections: List[Dict], min_detections: int = 5) -> List[Dict]:
    """
    Simple filter to remove tracks with too few detections.
    
    Parameters
    ----------
    tracked_detections : List[Dict]
        List of tracked detections
    min_detections : int
        Minimum number of detections to consider a track valid
        
    Returns
    -------
    List[Dict]
        Filtered detections
    """
    # Group detections by track ID
    tracks = defaultdict(list)
    for det in tracked_detections:
        tracks[det['ball_id']].append(det)
    
    # Filter tracks based on minimum detection count
    filtered_detections = []
    for track_id, track_detections in tracks.items():
        if len(track_detections) < min_detections:
            print(f"Removing track {track_id} (too few detections: {len(track_detections)})")
            continue
        
        filtered_detections.extend(track_detections)
        print(f"Keeping track {track_id} ({len(track_detections)} detections)")
    
    return filtered_detections


def process_detections(input_csv: str, 
                      output_csv: str,
                      max_distance: float = 150.0,
                      max_lost_frames: int = 120,
                      min_detections: int = 5) -> Dict:
    """
    Process ball detections to assign consistent IDs.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file with raw detections
    output_csv : str
        Path to output CSV file with tracked detections
    max_distance : float
        Maximum distance for track association
    max_lost_frames : int
        Maximum frames a track can be lost
    min_detections : int
        Minimum number of detections to consider a track valid
        
    Returns
    -------
    Dict
        Statistics about the processing
    """
    print(f"Loading detections from {input_csv}...")
    df = load_detections_from_csv(input_csv)
    
    if df.empty:
        print("No detections found in input file.")
        return {"total_detections": 0, "tracked_detections": 0, "unique_tracks": 0}
    
    print(f"Loaded {len(df)} detections from {df['frame_num'].nunique()} frames")
    
    # Initialize tracker
    tracker = SimpleBallTracker(max_distance=max_distance, max_lost_frames=max_lost_frames)
    
    # Process frame by frame
    frame_groups = df.groupby('frame_num')
    tracked_detections = []
    
    for frame_num, frame_detections in frame_groups:
        # Convert frame detections to the format expected by tracker
        detections = []
        for _, row in frame_detections.iterrows():
            detections.append({
                'bbox': [row['x1'], row['y1'], row['x2'], row['y2']],
                'confidence': row['confidence']
            })
        
        # Update tracker
        updated_detections = tracker.update(detections, frame_num)
        
        # Store tracked detections
        for det in updated_detections:
            tracked_detections.append({
                'frame_num': frame_num,
                'ball_id': det['track_id'],
                'x1': det['bbox'][0],
                'y1': det['bbox'][1],
                'x2': det['bbox'][2],
                'y2': det['bbox'][3],
                'confidence': det['confidence']
            })
    
    # Count unique tracks before filtering
    unique_tracks_before = len(set(det['ball_id'] for det in tracked_detections))
    print(f"Created {unique_tracks_before} unique tracks")
    
    # Filter tracks with too few detections
    print(f"Filtering tracks (min detections: {min_detections})...")
    filtered_detections = filter_tracks(tracked_detections, min_detections)
    
    # Write filtered detections to output CSV
    print(f"Writing filtered detections to {output_csv}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_num', 'ball_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
        
        for det in filtered_detections:
            writer.writerow([
                det['frame_num'],
                det['ball_id'],
                det['x1'],
                det['y1'],
                det['x2'],
                det['y2'],
                det['confidence']
            ])
    
    # Count unique tracks in filtered results
    unique_tracks_after_filter = len(set(det['ball_id'] for det in filtered_detections))
    
    stats = {
        "total_detections": len(df),
        "tracked_detections": len(tracked_detections),
        "filtered_detections": len(filtered_detections),
        "unique_tracks_before": unique_tracks_before,
        "unique_tracks_after": unique_tracks_after_filter
    }
    
    print(f"Processing complete!")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Tracked detections: {stats['tracked_detections']}")
    print(f"  Filtered detections: {stats['filtered_detections']}")
    print(f"  Unique tracks (before filter): {stats['unique_tracks_before']}")
    print(f"  Unique tracks (after filter): {stats['unique_tracks_after']}")
    
    return stats


def main():
    """Main function for ball tracking."""
    parser = argparse.ArgumentParser(description='Assign consistent IDs to ball detections')
    parser.add_argument('--input-dir', type=str, default='data/intermediate',
                       help='Input directory containing CSV files (default: data/intermediate)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for tracked CSV files (default: data/processed)')
    parser.add_argument('--max-distance', type=float, default=150.0,
                       help='Maximum distance for track association (default: 150.0)')
    parser.add_argument('--max-lost-frames', type=int, default=120,
                       help='Maximum frames a track can be lost (default: 120)')
    parser.add_argument('--min-detections', type=int, default=5,
                       help='Minimum number of detections to consider a track valid (default: 5)')
    parser.add_argument('--input-file', type=str, default=None,
                       help='Process only a specific CSV file (optional)')
    
    args = parser.parse_args()
    
    # Get list of CSV files to process
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        csv_files = [args.input_file]
    else:
        # Find all CSV files in input directory
        csv_files = list(Path(args.input_dir).glob('*_balls.csv'))
        csv_files = [str(f) for f in csv_files]
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    print(f"Max distance: {args.max_distance} pixels")
    print(f"Max lost frames: {args.max_lost_frames}")
    print(f"Min detections: {args.min_detections}")
    print("-" * 50)
    
    # Process each CSV file
    total_stats = {
        "total_detections": 0,
        "tracked_detections": 0,
        "filtered_detections": 0,
        "unique_tracks_before": 0,
        "unique_tracks_after": 0
    }
    
    for csv_file in csv_files:
        try:
            # Generate output filename
            input_name = Path(csv_file).stem
            output_file = os.path.join(args.output_dir, f"{input_name}_tracked.csv")
            
            print(f"\nProcessing {csv_file}...")
            stats = process_detections(
                csv_file, 
                output_file,
                args.max_distance,
                args.max_lost_frames,
                args.min_detections
            )
            
            # Accumulate statistics
            for key in total_stats:
                total_stats[key] += stats[key]
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print("-" * 50)
    print("Overall Statistics:")
    print(f"  Total detections: {total_stats['total_detections']}")
    print(f"  Tracked detections: {total_stats['tracked_detections']}")
    print(f"  Filtered detections: {total_stats['filtered_detections']}")
    print(f"  Unique tracks (before filter): {total_stats['unique_tracks_before']}")
    print(f"  Unique tracks (after filter): {total_stats['unique_tracks_after']}")


if __name__ == "__main__":
    main()