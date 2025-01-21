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
from ball_tracker import BallTracker
from padel_utils import get_feet_positions, draw_mini_court, draw_bboxes, draw_ball

# TODO: make auto-detection of points for perspective matrix. Add more point for fisheye correction

class PadelAnalyzer:
    class Model(Enum):
        FAST = 1
        MEDIUM = 2
        ACCURATE = 3

    def __init__(self, input_path, cam_name = None, output_video_path = None, output_csv_path = None, save_interval = 100, recalculate_matrix = False):
        """
        Initializes the PadelAnalyzer class.

        Parameters
        ----------
        input_path : str or int
            Path to the input video file or camera index.
        cam_name : str, optional
            Name of the camera. If not provided, the name will be the video name or the camera index.
        output_video_path : str, optional
            Path to save the output video. Defaults to "to_be_uploaded/{video_name}-analyzed.mp4". 
        output_csv_path : str, optional
            Path to save the output CSV files. Defaults to "output_data/{video_name}-period{1,2,3}.csv".
        save_interval : int, optional
            Interval at which data is saved. Defaults to 100.
        recalculate_matrix : bool, optional
            Flag to recalculate the perspective matrix. Defaults to False.

        Raises
        ------
        RuntimeError
            If the video/camera stream could not be opened.
        """
        self.cap = cv.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"PADEL ERROR: Could not open video/camera stream {input_path}.")
            
        # Set names
        if isinstance(input_path, str):
            self.file_opened = True
            self.video_name = os.path.basename(input_path)
            self.video_name = os.path.splitext(self.video_name)[0]
            self.cam_name = cam_name or self.video_name
        elif isinstance(input_path, int):
            self.file_opened = False
            self.video_name = str(input_path) + datetime.today().strftime('%Y-%m-%d-%H:%M')
            self.cam_name = cam_name or str(input_path)

        self.output_video_path = output_video_path or f"to_be_uploaded/{self.video_name}-analyzed.mp4"
        csv_name = output_csv_path or f"output_data/{self.video_name}"
        
        self.output_csv_paths = [csv_name + '-period1.csv', csv_name + '-period2.csv', csv_name + '-period3.csv']

        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv_paths[0]), exist_ok=True)

        # Load parmeters (fps, perspective and fisheye matrices)
        if recalculate_matrix:
            self.fps = self._calculate_fps()
            self.fisheye_matrix, self.distortion = self._calculate_fisheye_params()
            self.perspective_matrix = self._calculate_perspective_matrix()
        else:
            self.fps = self._load_fps()
            self.fisheye_matrix, self.distortion = self._load_fisheye_params()
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
        
    def process(self, model = Model.ACCURATE, debug = False):
        """Process all the frames of the video, detecting players and saving their positions in a CSV file.
        3 CSV files are created, one for each period of the game.
        
        Parameters
        ----------
        model : Model, optional
            The YOLO model to use for player detection. Options are Model.ACCURATE, Model.MEDIUM, and Model.FAST. Default is Model.ACCURATE.
        debug : bool, optional
            If True, additional debug information will be drawn on the video, such as player IDs and the mini court. Default is False.
        
        Raises
        ------
        FileNotFoundError
            If the video file is opened but the first frame cannot be read.
        
        Returns
        -------
        tuple
            A tuple containing:
            - output_video_path : str
                The path to the output video file.
            - fps : int
                The frames per second of the output video.
            - output_csv_paths : list
                A list of paths to the output CSV files containing player data.
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
        if model == PadelAnalyzer.Model.ACCURATE:
            model = 'models/yolov8x'
        elif model == PadelAnalyzer.Model.MEDIUM:
            model = 'models/yolov8m'
        elif model == PadelAnalyzer.Model.FAST:
            model = 'models/yolov8n'

        player_tracker = PlayerTracker(model_path=model)
        ball_tracker = BallTracker(model_path='models/ball_model.pt')
        all_frame_data = [None] * self.save_interval
        

        # ----- Main loop: one iteration per frame -----

        frame_num = 0
        period = 0  # 'tempo' in Italian
        longer_90 = False
        while True:
            success, frame = self.cap.read()
            if not success:
                print("End of video" if longer_90 else "End of video (before 90 minutes)")
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
                    longer_90 = True
                    print("Warning: last period longer than 30 minutes")
            
            # Detect players and ball
            detected_dict = player_tracker.detect(frame)
            balls = ball_tracker.detect(frame)

            # player_dict is a dictionary of {ID, namedtuple} the namedtuple is (bbox, positon)     player_dict = {ID, (bbox, position)}
            player_dict = self._calculate_positions(detected_dict)
            ball_positions = self._calculate_ball_positions(balls)
            
            # Choose best 4 players to show them in the video (but all are saved in the csv)
            best_player_dict = player_tracker.choose_best_players(player_dict)

            # Record and save data every save_interval frames
            frame_data = {
                'frame_num': frame_num,
                'balls': ball_positions,
                'players': player_dict #or {} # If player_dict is None, use an empty dictionary

            }
            # Instead of deleting, I'll just save starting all over
            all_frame_data[frame_num % self.save_interval] = frame_data
            if (frame_num+1) % self.save_interval == 0 and frame_num != 0:
                self._save_data_to_csv(all_frame_data, self.output_csv_paths[period])

            # Draw things and output video
            if debug:
                frame = draw_bboxes(frame, player_dict, show_id=True)
                frame = draw_ball(frame, balls)
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

        # Save the calculated fps
        path = self.cam_name + '-fps.txt'
        with open(path, 'w') as file:
            file.write(str(fps))

        return fps
    
    def _load_fisheye_params(self):
        path = self.cam_name + '-fisheye.txt'
        parameters = {}

        try:
            with open(path, 'r') as file:
                for line in file:
                    # Split the line by '=' to separate key and value
                    key, value = map(str.strip, line.strip().split('='))
                    parameters[key] = float(value)  # Convert value to float
        except FileNotFoundError:
            return self._calculate_fisheye_params()

        if not parameters:
            raise ValueError("Error: No parameters found in the file.")

        fx = parameters['fx']
        fy = parameters['fy']
        cx = parameters['cx']
        cy = parameters['cy']
        k1 = parameters['k1']
        k2 = parameters['k2']
        p1 = parameters['p1']
        p2 = parameters['p2']
        k3 = parameters['k3']

        mtx = np.array(
                        [[fx, 0., cx],
                         [0., fy, cy],
                         [0., 0., 1.]])   
        dist = np.array([[k1, k2, p1, p2, k3]])

        return mtx, dist
    
    def _calculate_fisheye_params(self):
        # TODO: Calculate parameters automatically given some points or with checkerboard

        from gui_calib import Fisheye

        random.seed(time.time())
        totalFrame = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) - 2
        self.cap.set(cv.CAP_PROP_POS_FRAMES, random.randint(0, totalFrame))
        ret, img = self.cap.read()
    
        path = self.cam_name + '-fisheye.txt'
        fisheye = Fisheye(img)
        mtx, dist = fisheye.fisheye_gui(save_path=path)

        return mtx, dist

    def _load_perspective_matrix(self):
        path = self.cam_name + '-matrix.txt'
        try:
            matrix = np.loadtxt(path)
        except FileNotFoundError:
            matrix = self._calculate_perspective_matrix()
        return matrix

    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.mousePosition = (x, y)
        elif event == cv.EVENT_RBUTTONDOWN:
            self.mousePosition = (-2, -2)   # TODO: this does not work. Maybe change it to a key pressed
    
    def _calculate_perspective_matrix(self):
        # TODO: use undistorted image (with fisheye correction) to calculate perspective matrix
        
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

        # Save the calculated matrix
        path = self.cam_name + '-matrix.txt'
        np.savetxt(path, perspMat)

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
    
    def _calculate_ball_positions(self, balls):
        """ 
        Calculate positions (in meter) of balls.
        Given the detected balls, return a list of positions.
        """
        if not balls:
            return []
        
        balls = np.array(balls)
        balls = balls[:, :2] + (balls[:, 2:] - balls[:, :2]) / 2
        balls = balls.reshape(-1, 1, 2)
        positions = cv.perspectiveTransform(balls, self.perspective_matrix).reshape(-1, 2)
        
        return positions

    def _save_data_to_csv(self, data, path):
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for frame_data in data:
                row = [frame_data['frame_num'], frame_data['balls']]
                for player_id, player_info in frame_data['players'].items():
                    row.append(player_id)
                    row.append(player_info.position)
                writer.writerow(row)
