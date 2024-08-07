import os
import random
import time
from collections import namedtuple
from datetime import datetime
from enum import Enum
# TODO import logging

import cv2 as cv
import numpy as np
import csv

from player_tracker import PlayerTracker
from padel_utils import get_feet_positions, draw_mini_court, get_distance, draw_stats, ensure_directory_exists, draw_bboxes

# TODO: make auto-detection of points for perspective matrix

class PadelAnalyzer:
    class Method(Enum):
        FAST = 1
        MEDIUM = 2
        ACCURATE = 3

    def __init__(self, input_path, cam_name = None, output_video_path = None, output_csv_path = None, save_interval = 100, recalculate_matrix = False):
        self.cap = cv.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"PADEL ERROR: Could not open video/camera stream {input_path}.")
            
        # Set names
        if isinstance(input_path, str):
            self.file_opened = True
            self.video_name = input_path.replace('/','-').replace("\\",'-')
            self.video_name = os.path.splitext(self.video_name)[0]
            self.cam_name = cam_name or self.video_name
        elif isinstance(input_path, int):
            self.file_opened = False
            self.video_name = str(input_path) + datetime.today().strftime('%Y-%m-%d-%H:%M')
            if cam_name == None:
                self.cam_name = str(input_path)

        self.output_video_path = output_video_path or f"to_be_uploaded/{self.video_name}-analyzed.mp4"
        csv_name = output_csv_path or f"output_data/{self.video_name}"
        
        self.output_csv_paths = [csv_name + '-period1.csv', csv_name + '-period2.csv', csv_name + '-period3.csv']

        ensure_directory_exists(self.output_video_path)    
        ensure_directory_exists(self.output_csv_paths[0])

        # Load fps and matrix
        if recalculate_matrix:
            self.fps = self._calculate_fps()
            self.perspective_matrix = self._calculate_perspective_matrix()
        else:
            self.fps = self._load_fps()
            self.perspective_matrix = self._load_perspective_matrix()
        
        # how many frames to average to calculate average velocity (2*fps = 2 seconds)
        self.mean_interval = int(1*self.fps)
        
        # Data saving structure
        self.save_interval = save_interval
        # if (self.mean_interval >= self.save_interval):  #improbable but better to check
        #     self.save_interval = self.mean_interval + 1

        for csv_path in self.output_csv_paths:
            with open(csv_path, 'w', newline='') as csvfile:
                pass    # Create the file (erasing it if it already exists)


    def process_frame(self):
        # TO SEE: keep it or remove it?
        return
        
    def process_all(self, method = Method.ACCURATE, debug = False):
        """
        Process all the frames of the video. 
        method: PadelAnalyzer.Method.ACCURATE, PadelAnalyzer.Method.MEDIUM, PadelAnalyzer.Method.FAST
        debug: bool
        Return: the path of the output video, its fps, and the path of the csv files (results).
        """

        # whenever process_all is called, re-start from beginning of video
        if self.file_opened:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Take first frame
        success, first_frame = self.cap.read()
        if not success:
            raise FileNotFoundError(f"PADEL ERROR: Cap is opened, but couldn't read first frame.")
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(self.output_video_path, fourcc, self.fps, (first_frame.shape[1], first_frame.shape[0]))
        
        # Choosing model
        if method == PadelAnalyzer.Method.ACCURATE:
            model = 'yolov8x'
        elif method == PadelAnalyzer.Method.MEDIUM:
            model = 'yolov8m'
        elif method == PadelAnalyzer.Method.FAST:
            model = 'yolov8n'

        player_tracker = PlayerTracker(model_path=model)
        all_frame_data = [None] * self.save_interval
        
        # --- Main loop: one iteration per frame ---
        frame_num = 0
        period = 0  # 'tempo' in Italian
        while True:
            success, frame = self.cap.read()
            if not success:
                print("End of video (before 30 minutes)")
                break
            
            if frame_num == self.fps*60*30: # after 30 minutes..
                if (period < 2):            #..but only if it's not already the last period
                    print("End 30 minutes")
                    player_tracker = PlayerTracker(model_path=model)    # Restart the tracking
                    #Save data remained in buffer and clear data
                    self._save_data_to_csv(all_frame_data[:(frame_num%self.save_interval)], self.output_csv_paths[period])
                    all_frame_data = [None] * self.save_interval
                    frame_num = 0
                    period += 1
                else:   
                    print("Warning: last period longer than 30 minutes")
            
            # Detect players
            detected_dict = player_tracker.detect(frame)
            # player_dict is a dictionary of {ID, namedtuple} the namedtuple is (bbox, positon)     player_dict = {ID, (bbox, position)}
            player_dict = self._calculate_positions(detected_dict)
            # Choose best 4 players to show them in the video (but all are saved in the csv)
            best_player_dict = player_tracker.choose_players(player_dict)

            # Record and save data every save_interval frames
            frame_data = frame_data = {
                'frame_num': frame_num,
                'players': player_dict #or {} # If player_dict is None, use an empty dictionary
            }
            # Instead of deleting, I'll just save starting all over
            all_frame_data[frame_num % self.save_interval] = frame_data
            if (frame_num+1) % self.save_interval == 0 and frame_num != 0:
                self._save_data_to_csv(all_frame_data, self.output_csv_paths[period])

            # Draw things and output video
            if debug:
                frame = draw_bboxes(frame, player_dict, show_id=True)
                frame = draw_mini_court(frame, player_dict)
            else:
                frame = draw_bboxes(frame, best_player_dict, show_id=False)
                frame = draw_mini_court(frame, best_player_dict)
            # frame = draw_stats(frame, frame_data)     Can't draw stats because only position is saved (other data to be calculated during postprocess)

            out.write(frame)
        
            frame_num += 1

        self.cap.release()
        out.release()

        # Save data remained in buffer
        self._save_data_to_csv(all_frame_data[:(frame_num%self.save_interval)], self.output_csv_paths[period])

        return self.output_video_path, self.fps, self.output_csv_paths


    # --Helper "private" functions--

    def _load_fps(self):
        path = self.cam_name + '-fps.txt'
        try:
            with open(path, 'r') as file:
                content = file.read().strip()
                return float(content)
        except FileNotFoundError:
            fps = self._calculate_fps()
            with open(path, 'w') as file:
                file.write(str(fps))
            return fps
        except ValueError:
            print(f"PADEL ERROR: File path {path} does not contain a valid number, maybe it's corructed. Delete it in order to re-create it.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def _calculate_fps(self):
        fps = self.cap.get(cv.CAP_PROP_FPS)
        if fps != 0:
            return fps
        
        #If the property is 0, try to calculate it in a different way
        num_frames = 60
        start = time.time()
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
        end = time.time()
        if self.file_opened:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        fps = num_frames / (end-start)
        return fps

    def _load_perspective_matrix(self):
        path = self.cam_name + '-matrix.txt'
        try:
            matrix = np.loadtxt(path)
        except FileNotFoundError:
            matrix = self._calculate_perspective_matrix()
            np.savetxt(path, matrix)
        return matrix

    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.mousePosition = (x, y)
        elif event == cv.EVENT_RBUTTONDOWN:
            self.mousePosition = (-2, -2)   # TODO: this does not work. Maybe change it to a key pressed
    
    def _calculate_perspective_matrix(self):
        self.mousePosition = (-1,-1)
        winName = "Click on points indicated in green. Use WASD to move last cross. Right click to remove it. Press space to confirm."
        cv.namedWindow(winName)
        cv.setMouseCallback(winName, self._onMouse)

        angles = []  # angles of the field

        if self.file_opened:
            print("--- Calculating perspective ---\nClick on the 4 indicated points. Press 'n' to show another (random) frame. Press 'space' to confirm. Press 'q' to abort")
        else:
            print("--- Calculating perspective ---\nClick on the 4 indicated points. Press 'space' to confirm. Press 'q' to abort")

        random.seed(time.time())
        totalFrame = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) - 2
        key = -1
        count = 0
        while key != ord(' '):  # space key
            key = -1
            success, original = self.cap.read()
            if self.file_opened:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, random.randint(0, totalFrame))
                success, original = self.cap.read()
            while self.file_opened and key != ord('n') and (key != 32 or count < 4):
                frame = original.copy()
                if self.mousePosition[0] == -2:  # right click on mouse
                    if count > 0:
                        count -= 1
                    self.mousePosition = (-1, -1)
                elif self.mousePosition[0] != -1:  # if mouse has been clicked
                    if count >= 4:  # if user is trying to add fifth point, break
                        key = ord(' ')  # space key
                        break
                    angles.append(self.mousePosition)
                    count += 1
                    self.mousePosition = (-1, -1)

                key = cv.waitKey(5)
                if key == ord('q'):
                    cv.destroyWindow(winName)
                    raise RuntimeError("PADEL NOTE: execution aborted by user.")
                if count > 0:
                    if key == ord('a'):  # left
                        angles[-1] = (angles[-1][0] -1, angles[-1][1]   )
                    elif key == ord('d'):  # right
                        angles[-1] = (angles[-1][0] +1, angles[-1][1]   )
                    elif key == ord('w'):  # up
                        angles[-1] = (angles[-1][0]   , angles[-1][1] -1)
                    elif key == ord('s'):  # down
                        angles[-1] = (angles[-1][0]   , angles[-1][1] +1)

                for angle in angles:
                    cv.drawMarker(frame, angle, (0, 0, 255), cv.MARKER_TILTED_CROSS, 20, 1)

                scale = 10
                offset = 20
                cv.rectangle(frame, (offset, offset), (10 * scale + offset, 20 * scale + offset), (129, 94, 61), -1)
                cv.line(frame, (offset, 3 * scale + offset), (10 * scale + offset, 3 * scale + offset), (255, 255, 255), 1)
                cv.line(frame, (offset, 17 * scale + offset), (10 * scale + offset, 17 * scale + offset), (255, 255, 255), 1)
                cv.line(frame, (5 * scale + offset, int(2.7 * scale + offset)), (5 * scale + offset, int(17.3 * scale + offset)), (255, 255, 255), 1)
                cv.line(frame, (offset, 10 * scale + offset), (10 * scale + offset, 10 * scale + offset), (0, 0, 255), 2)
                cv.drawMarker(frame, (offset, offset), (0, 255, 0), cv.MARKER_TILTED_CROSS, scale, 2)
                cv.drawMarker(frame, (10 * scale + offset, offset), (0, 255, 0), cv.MARKER_TILTED_CROSS, scale, 2)
                cv.drawMarker(frame, (offset, 17 * scale + offset), (0, 255, 0), cv.MARKER_TILTED_CROSS, scale, 2)
                cv.drawMarker(frame, (10 * scale + offset, 17 * scale + offset), (0, 255, 0), cv.MARKER_TILTED_CROSS, scale, 2)

                cv.imshow(winName, frame)

        cv.destroyWindow(winName)

        angles.sort(key=lambda pt: pt[1])
        if angles[0][0] > angles[1][0]:
            angles[0], angles[1] = angles[1], angles[0]
        if angles[2][0] < angles[3][0]:
            angles[2], angles[3] = angles[3], angles[2]

        rect = np.float32([[0, 0], [10, 0], [10, 17], [0, 17]])
        perspMat = cv.getPerspectiveTransform(np.float32(angles), rect)

        if self.file_opened:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        return perspMat
    
    def _calculate_positions(self, detected_dict):
        """ 
        Calculate positions (in meter) of players.
        Given the detected dictionary {ID, bbox}, return {ID, (bbox, position)}
        """
        if not detected_dict:
            return {}
        
        bboxes = [box for box in detected_dict.values()]
        # Reshape feet positions to the required shape (n, 1, 2) (the extra 1 required from perspectiveTransform)
        feet = get_feet_positions(bboxes).reshape(-1, 1, 2) 
        positions = cv.perspectiveTransform(feet, self.perspective_matrix).reshape(-1,2)    # reshape back to (n,2)
        PlayerInfo = namedtuple('PlayerInfo', ['bbox', 'position'])

        return {id: PlayerInfo(detected_dict[id], positions[i]) for i, id in enumerate(detected_dict)}

    def _record_frame_data(self, frame_num, player_dict = {}):
        null_pos = np.array([0.0, 0.0])
        frame_data = {
            'frame_num': frame_num,
            'player_1_id': '',
            'player_1_position': null_pos,
            'player_1_distance': 0,
            'player_1_speed': 0,
            'player_2_id': '',
            'player_2_position': null_pos,
            'player_2_distance': 0,
            'player_2_speed': 0,
            'player_3_id': '',
            'player_3_position': null_pos,
            'player_3_distance': 0,
            'player_3_speed': 0,
            'player_4_id': '',
            'player_4_position': null_pos,
            'player_4_distance': 0,
            'player_4_speed': 0
        }

        if player_dict is None:
            return frame_data

        for i, (player_id, player_info) in enumerate(player_dict.items()):
            frame_data[f'player_{i + 1}_id'] = player_id
            frame_data[f'player_{i + 1}_position'] = player_info.position

            if frame_num > self.mean_interval:    #calculate speed and distances only after having accumulated enogh data
                # data of mean_interval frames ago
                last_data = self.all_frame_data[frame_num%self.save_interval - self.mean_interval]  #if negative roll back up
                distance = get_distance(player_info.position, last_data[f'player_{i+1}_position'])
                
                # add distance only between each n frame, oterwise I'll add to much (non ho voglia di spiegare)
                if frame_num % self.mean_interval == 0:
                    cumulative_distance = distance + last_data[f'player_{i+1}_distance']
                else:
                    # take frame - fram%mean_interval
                    last_distance = self.all_frame_data[(frame_num - (frame_num%self.mean_interval))%self.save_interval]
                    cumulative_distance = last_distance[f'player_{i+1}_distance']
                
                frame_data[f'player_{i + 1}_distance'] = cumulative_distance

                dt = self.mean_interval/self.fps
                frame_data[f'player_{i + 1}_speed'] = 3.6*distance/dt 

        return frame_data

    def _save_data_to_csv(self, data, path):
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for frame_data in data:
                row = [frame_data['frame_num']]
                for player_id, player_info in frame_data['players'].items():
                    row.append(player_id)
                    row.append(player_info.position)
                writer.writerow(row)
