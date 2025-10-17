from ultralytics import YOLO
import pickle
import cv2
import os

model = YOLO("weights/two_classes_best.pt")

# Prepare to write video with plotted tracks
video_path = "data/raw/video_test.mp4"
output_path = "data/processed/video_test_tracked_botsort.mp4"

# Open original video to get properties
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

from tqdm import tqdm

# Get total number of frames for the progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
with tqdm(total=total_frames, desc='Tracking frames') as pbar:
    for result in model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="custom_botsort.yaml",
        show=False,
        save=False,
        verbose=False
    ):
        # result.plot() returns a numpy frame with drawings
        plotted_frame = result.plot()
        out.write(plotted_frame)
        frame_idx += 1
        pbar.update(1)

out.release()
cap.release()
print("Tracking results have been saved to data/processed/video_test_tracking_results.pkl")
print(f"Tracked video with plotted results has been saved to {output_path}")
