""" 
What I noticed is:
- threshold = 12 results in too many detections, and the ball is not
  even light bright green (more a green-ish gray).
  This even with min_area=4
- min_area = 0 results in a tremendous quantity of possible balls
  (at least with threshold <= 46)
- t=21,a=4 is way better, but still there are too many possible balls
- t=26,a=4 even better. Maybe still too much
- t=46,a=4 seems to detect too few times the ball
- t=46,a=1 seems better, but still few balls
- t=44 is better. Almost good (still requires dynamics). a=1 better than a=4
- t=37,a=4 detects ever so slightly better, with lot of noise
  (I think it's not worth it)
"""

import numpy as np
from debug_utils import load_data
from padel_utils import triangulate_points, draw_mini_court
import csv
from tqdm import tqdm
import cv2
import os
import re
import argparse

def generate3Dcsv():
    """Generate triangulated point csv for each value of parameter"""

    O1 = np.array([5, 20.325, 2.635])
    O2 = np.array([5, -0.325, 2.635])

    for thresh in range(59, 9, -1): #from 10 to 60 but in reverse
        for min_area in range(4,-1,-1):   #from 0 to 5 in reverse
            path1 = f"output_data/new1_t={thresh},a={min_area}.csv"
            path2 = f"output_data/new2_t={thresh},a={min_area}.csv"
            
            try:
                data1 = load_data(path1)
                data2 = load_data(path2)
            except FileNotFoundError:
                print(f"Not found t={thresh}, a={min_area}")
                continue
            
            print(f"Found: t={thresh}, a={min_area}")

            out_path = f"output_data/triangulated_new_t={thresh},a={min_area}.csv"

            with open(out_path, "w"):
                pass

            writer = csv.writer(open(out_path, "w"))

            N = min(len(data1), len(data2))
            for frame in range(N):
                row = [frame]
                points, errors = triangulate_points(O1, O2, data1[frame], data2[frame])
                if points is not None:
                    for p, e in zip(points, errors):
                        # Discard:
                        # - points below the surface of the field
                        # - points with a very high error
                        # - points outside the field
                        if p[2] < -0.5:
                            continue
                        if e > 10:
                            continue
                        if p[0] < 0.5 or p[0] > 10.5 or p[1] < 0.5 or p[1] > 20.5:
                            continue
                        
                        row.append(p)
                        row.append(e)
                
                writer.writerow(row)


def create_videos(output_dir = "output_videos", start_thresh = 10):
    """Create the videos"""

    os.makedirs(output_dir, exist_ok=True)

    for thresh in range(start_thresh, 60):
        for min_area in range(0, 5):
            in_path = f"output_data/triangulated_new_t={thresh},a={min_area}.csv"

            data = []
            try:
                with open(in_path, "r") as f:
                    print(f"Found: t={thresh}, a={min_area}. Creating video...")
                    reader = csv.reader(f)

                    for row in reader:
                        frame_num = int(row[0])

                        detections = []
                        for i in range(1, len(row), 2):
                            detection_pos_str = row[i].strip('[] ')
                            detection_pos = [float(coord) for coord in detection_pos_str.split()]
                            detection_error = float(row[i+1])
                            
                            detections.append((detection_pos, detection_error))
                        # frame_data = (frame_num, detections)
                        # data.append(frame_data)
                        data.append(detections)
            except:
                print(f"Not found: t={thresh}, a={min_area}")
                continue

            video_path = "input_videos/Nuova qualità/video1.mp4"
            out_path = f"{output_dir}/triangulated_new_t={thresh},a={min_area}.mp4"
            skip_frame = 0
            sync_frame = 0

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(fps)
            print(size)
            print(f"Data length: {len(data)}")

            ret, frame = cap.read()
            print(frame.shape[1], frame.shape[0])

            for _ in range(skip_frame):
                cap.read()

            video_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            for frame_num in range(len(data)):
                for _ in range(sync_frame):
                    cap.read()
                ret, frame = cap.read()
                if not ret:
                    print(f"Breaking at {frame_num}")
                    break

                detections = data[frame_num]
                # positions = [d[0] for d in detections]
                # errors = [d[1] for d in detections]

                frame = draw_mini_court(frame, None, detections)

                video_out.write(frame)

                #cv2.imshow("Frame", frame)
                #if cv2.waitKey(0) & 0xFF == ord('q'):
                #    break

            cap.release()
            print("Releasing...")
            video_out.release()
            #cv2.destroyAllWindows()

def delete_used(folder_path = "output_data"):
    """
    Scan a folder, if there is file triangulated_new_t={thresh},a={min_area}.csv, 
    then remove file new1_t={thresh},a={min_area}.csv and remove file new2_t={thresh},a={min_area}.csv
    """

    # Regex patterns for matching files
    triangulated_pattern = re.compile(r"triangulated_new_t=(\d+\.?\d*),a=(\d+\.?\d*)\.csv")
    new1_pattern = "new1_t={thresh},a={min_area}.csv"
    new2_pattern = "new2_t={thresh},a={min_area}.csv"

    # Store thresholds and areas from triangulated files
    thresh_area_pairs = set()

    # Scan the folder and collect matching patterns
    for filename in os.listdir(folder_path):
        match = triangulated_pattern.match(filename)
        if match:
            thresh, min_area = match.groups()
            thresh_area_pairs.add((thresh, min_area))

    # Delete corresponding "new1" files
    for thresh, min_area in thresh_area_pairs:
        file_to_delete = new1_pattern.format(thresh=thresh, min_area=min_area)
        file_path = os.path.join(folder_path, file_to_delete)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_to_delete}")
        else:
            print(f"File not found: {file_to_delete}")

        file_to_delete = new2_pattern.format(thresh=thresh, min_area=min_area)
        file_path = os.path.join(folder_path, file_to_delete)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_to_delete}")
        else:
            print(f"File not found: {file_to_delete}")

def delete_finished(folder_path="output_data", video_folder="output_videos"):
    """
    Scan the folder "output_data" for files named "triangulated_new_t={thresh},a={min_area}.csv".
    If a corresponding video "triangulated_new_t={thresh},a={min_area}.mp4" exists in "output_videos",
    remove the CSV file from "output_data".
    """
    # Regex pattern for matching files
    triangulated_pattern = re.compile(r"triangulated_new_t=(\d+\.?\d*),a=(\d+\.?\d*)\.csv")
    video_pattern = "triangulated_new_t={thresh},a={min_area}.mp4"
    
    # Store thresholds and areas from triangulated CSV files
    thresh_area_pairs = set()
    
    # Scan the folder and collect matching patterns
    for filename in os.listdir(folder_path):
        match = triangulated_pattern.match(filename)
        if match:
            thresh, min_area = match.groups()
            thresh_area_pairs.add((thresh, min_area))
    
    # Check for corresponding videos and delete CSVs
    for thresh, min_area in thresh_area_pairs:
        video_file = video_pattern.format(thresh=thresh, min_area=min_area)
        video_path = os.path.join(video_folder, video_file)
        csv_file = f"triangulated_new_t={thresh},a={min_area}.csv"
        csv_path = os.path.join(folder_path, csv_file)
        
        if os.path.exists(video_path):
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"Deleted: {csv_file}")
            else:
                print(f"File not found: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', action=argparse.BooleanOptionalAction, help='generate 3D csv')
    parser.add_argument('-d', '--delete', action=argparse.BooleanOptionalAction, help='delete used 2D csv files')
    parser.add_argument('-c', '--create', action=argparse.BooleanOptionalAction, help='create videos from 3D csv')
    parser.add_argument('-e', '--end', action=argparse.BooleanOptionalAction, help='delete used 3D csv files')

    
    args = parser.parse_args()

    if args.generate:
        generate3Dcsv()
    elif args.delete:
        delete_used()
    elif args.create:
        create_videos(start_thresh=22)
    elif args.end:
        delete_finished()
    else:
        print("Please select an option (-h) for help")