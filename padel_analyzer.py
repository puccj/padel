import os
import random
import time
from collections import namedtuple
from datetime import datetime
from enum import Enum
# TODO import logging

import cv2
import numpy as np
import csv

from player_tracker import PlayerTracker
from ball_tracker import BallTracker
from padel_utils import get_feet_positions, load_fisheye_params, transform_points, draw_bboxes, draw_balls, draw_mini_court

# TODO: make auto-detection of points for perspective matrix. Add more point for fisheye correction

class PadelAnalyzer:

    def __init__(self, input_path, cam_name=None, cam_type=None, second_camera=False, output_video_path=None, output_csv_path=None, save_interval=100, recalculate=False):
        """
        Initializes the PadelAnalyzer class.

        Parameters
        ----------
        input_path : str or int
            Path to the input video file or camera index.
        cam_name : str, optional
            Name of the camera. If not provided, the name will be the video name or the camera index.
        cam_type : str, optional
            Type of the camera. Ideally it would be the name of the camera model or an alias of it.
            It is used to load the fisheye parameters. If set to None (default), no fisheye correction will be applied.
        second_camera : bool, optional
            Flag to indicate if the camera is the second one. Defaults to False.
        output_video_path : str, optional
            Path to save the output video. Defaults to "to_be_uploaded/{video_name}-analyzed.mp4". 
        output_csv_path : str, optional
            Path to save the output CSV files. Defaults to "output_data/{video_name}-period{1,2,3}.csv".
        save_interval : int, optional
            Interval at which data is saved. Defaults to 100.
        recalculate : bool, optional
            Flag to recalculate fps, and perspective matrix. Defaults to False.

        Raises
        ------
        RuntimeError
            If the video/camera stream could not be opened.
        """
        self.cap = cv2.VideoCapture(input_path)
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

        self.mousePosition = None  # mouse position for debug mode

        # Load parameters (fps, perspective and fisheye matrices)
        
        if cam_type is None:
            self.K, self.D = None, None
        else:
            fisheye_path = os.path.join('parameters', cam_type + '-fisheye.txt')
            try:
                self.K, self.D = load_fisheye_params(fisheye_path)  # fisheye matrix and distortion coefficients (K, D) are never re-calculated
            except FileNotFoundError:
                print("Fisheye parameters not found. You can follow the calibrate.py script to calculate them or continue to determine them manually.")
                print("\n--- Calculating fisheye parameters ---\n")
                from gui_calib import Fisheye
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
                ret, img = self.cap.read()
                fisheye = Fisheye(img)
                self.K, self.D = fisheye.fisheye_gui(save_path=fisheye_path)

        if recalculate:
            self.fps = self._calculate_fps()
            self.H = self._calculate_perspective_matrix(second_camera)
        else:
            self.fps = self._load_fps()
            self.H = self._load_perspective_matrix(second_camera)
        
        # how many frames to average to calculate average velocity (2*fps = 2 seconds)
        self.mean_interval = int(1*self.fps)
        
        # Data saving structure
        self.save_interval = save_interval
        # if (self.mean_interval >= self.save_interval):  #improbable but better to check
        #     self.save_interval = self.mean_interval + 1

        for csv_path in self.output_csv_paths:
            with open(csv_path, 'w', newline='') as csvfile:
                pass    # Create the file (erasing it if it already exists)

    def process(self, model='models/yolov11x.pt', ball_model='models/ball-11x-1607.pt', show=False, debug=False, mini_court=True):
        """
        Process all the frames of the video, detecting players and saving their positions in a CSV file.
        3 CSV files are created, one for each period of the game.
        
        Parameters
        ----------
        model : Model, optional
            The YOLO model to use for player detection. Default is 'models/yolov11x.pt'.
        ball_model : str, optional
            Path to the YOLO model for ball detection. Default is 'models/ball-11x-1607.pt'.
        show : bool, optional
            If True, the video will be displayed while processing. Default is False.
        debug : bool, optional
            If True, additional debug information will be drawn on the video, such as player IDs and the mini court. Default is False.
        mini_court : bool, optional
            If True, the mini court will be drawn on the video. Default is True.
        
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
        
        Raises
        ------
        FileNotFoundError
            If the video file is opened but the first frame cannot be read.
        """        

        # whenever process_all is called, re-start from beginning of video
        if self.file_opened:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Take first frame
        success, first_frame = self.cap.read()
        if not success:
            raise FileNotFoundError(f"PADEL ERROR: Cap is opened, but couldn't read first frame.")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (first_frame.shape[1], first_frame.shape[0]))

        player_tracker = PlayerTracker(model)
        ball_tracker = BallTracker(ball_model)
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
            balls_bbox = np.array([b[0] for b in balls])         # Extract bounding boxes from detections

            # player_dict is a dictionary of {ID, namedtuple} the namedtuple is (bbox, positon)     player_dict = {ID, (bbox, position)}
            player_dict = self._calculate_positions(detected_dict)
            ball_positions = self._calculate_ball_positions(balls_bbox)

            # Record and save data every save_interval frames
            frame_data = {
                'frame_num': frame_num,
                'balls': ball_positions,
                'players': player_dict # Note that player_dict could be None if no players are detected
            }
            # Instead of deleting, I'll just save starting all over
            all_frame_data[frame_num % self.save_interval] = frame_data
            if (frame_num+1) % self.save_interval == 0 and frame_num != 0:
                self._save_data_to_csv(all_frame_data, self.output_csv_paths[period])

            # Draw things and output video
            if debug:
                frame = draw_bboxes(frame, player_dict, show_id=True)
                frame = draw_balls(frame, balls_bbox)
                if show and self.mousePosition is not None:
                    mouse = transform_points(np.float32([self.mousePosition]), self.K, self.D, self.H)[0]
                    cv2.namedWindow('Video')
                    cv2.setMouseCallback('Video', self._onMouse)
                else:
                    mouse = None
                if mini_court:
                    frame = draw_mini_court(frame, player_dict, mouse)
            else:
                best_player_dict = player_tracker.choose_best_players(player_dict)
                frame = draw_bboxes(frame, best_player_dict, show_id=False)
                if mini_court:
                    frame = draw_mini_court(frame, best_player_dict)
            # frame = draw_stats(frame, frame_data)     Can't draw stats because only position is saved (other data to be calculated during postprocess)

            if show:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.write(frame)
        
            frame_num += 1

        # ----- End of main loop -----

        cv2.destroyAllWindows()
        self.cap.release()
        out.release()

        # Save data remained in buffer
        self._save_data_to_csv(all_frame_data[:(frame_num%self.save_interval)], self.output_csv_paths[period])

        return self.output_video_path, self.fps, self.output_csv_paths


    # --Helper "private" functions--

    def _load_fps(self):
        path = os.path.join('parameters', self.cam_name + '-fps.txt')
        try:
            with open(path, 'r') as file:
                content = file.read().strip()
                return float(content)
        except FileNotFoundError:
            fps = self._calculate_fps()
            return fps
        except ValueError:
            raise ValueError(f"PADEL ERROR: File path {path} does not contain a valid number, maybe it's corructed. Delete it manually to re-create it.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def _calculate_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
        
            #If the property is 0, try to calculate it in a different way
            num_frames = 60
            start = time.time()
            for _ in range(num_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
            end = time.time()
            if self.file_opened:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = num_frames / (end-start)

        # Save the calculated fps
        path = os.path.join('parameters', self.cam_name + '-fps.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            file.write(str(fps))

        return fps

    def _load_perspective_matrix(self, second_camera=False):
        path = os.path.join('parameters', self.cam_name + '-perspective.txt')
        try:
            matrix = np.loadtxt(path)
        except FileNotFoundError:
            matrix = self._calculate_perspective_matrix(second_camera)
        return matrix

    def _onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mousePosition = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mousePosition = (-2, -2)   # TODO: this (right click to remove cross) does not work. Maybe change it to a key pressed
    
    def _calculate_perspective_matrix(self, second_camera=False):
        self.mousePosition = None
        winName = "Click on points indicated in green. Use WASD to move last cross. Right click to remove it. Press space to confirm."
        cv2.namedWindow(winName)
        cv2.setMouseCallback(winName, self._onMouse)

        angles = []  # angles of the field

        if self.file_opened:
            print("\n--- Calculating perspective ---\n\nClick on the 4 indicated points. Press 'n' to show another (random) frame. Press 'space' to confirm. Press 'q' to abort")
        else:
            print("\n--- Calculating perspective ---\n\nClick on the 4 indicated points. Press 'space' to confirm. Press 'q' to abort")

        random.seed(time.time())
        totalFrame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
        key = -1
        count = 0
        while key != ord(' '):  # space key
            key = -1
            success, original = self.cap.read()
            if self.file_opened:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, totalFrame))
                success, original = self.cap.read()
            while self.file_opened and key != ord('n') and (key != 32 or count < 4):
                # Use undistorted image
                if self.K is not None and self.D is not None:
                    size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, size, cv2.CV_16SC2)
                    frame = cv2.remap(original, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    # frame = cv2.undistort(original, self.fisheye_matrix, self.distortion, None, None)
                else:
                    frame = original.copy()

                # #TODO: use a key instead of mouse right click
                # if self.mousePosition[0] == -2:  # right click on mouse 
                #     if count > 0:
                #         count -= 1
                #     self.mousePosition = (-1, -1)
                if self.mousePosition is not None:  # if mouse has been clicked
                    if count >= 4:  # if user is trying to add fifth point, break
                        key = ord(' ')  # space key
                        break
                    angles.append(self.mousePosition)
                    count += 1
                    self.mousePosition = None

                key = cv2.waitKey(5)
                if key == ord('q'):
                    cv2.destroyWindow(winName)
                    raise RuntimeError("Execution aborted by user during perspective calculation (Q key pressed).")
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
                    cv2.drawMarker(frame, angle, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 1)

                scale = 10
                offset = 20
                cv2.rectangle(frame, (offset, offset), (10 * scale + offset, 20 * scale + offset), (129, 94, 61), -1)
                cv2.line(frame, (offset, 3 * scale + offset), (10 * scale + offset, 3 * scale + offset), (255, 255, 255), 1)
                cv2.line(frame, (offset, 17 * scale + offset), (10 * scale + offset, 17 * scale + offset), (255, 255, 255), 1)
                cv2.line(frame, (5 * scale + offset, int(2.7 * scale + offset)), (5 * scale + offset, int(17.3 * scale + offset)), (255, 255, 255), 1)
                cv2.line(frame, (offset, 10 * scale + offset), (10 * scale + offset, 10 * scale + offset), (0, 0, 255), 2)
                cv2.drawMarker(frame, (offset, offset), (0, 255, 0), cv2.MARKER_TILTED_CROSS, scale, 2)
                cv2.drawMarker(frame, (10 * scale + offset, offset), (0, 255, 0), cv2.MARKER_TILTED_CROSS, scale, 2)
                cv2.drawMarker(frame, (offset, 17 * scale + offset), (0, 255, 0), cv2.MARKER_TILTED_CROSS, scale, 2)
                cv2.drawMarker(frame, (10 * scale + offset, 17 * scale + offset), (0, 255, 0), cv2.MARKER_TILTED_CROSS, scale, 2)

                cv2.imshow(winName, frame)

        cv2.destroyWindow(winName)

        angles.sort(key=lambda pt: pt[1])
        if angles[0][0] > angles[1][0]:
            angles[0], angles[1] = angles[1], angles[0]
        if angles[2][0] < angles[3][0]:
            angles[2], angles[3] = angles[3], angles[2]

        if second_camera:
            rect = np.float32([[10,20], [0,20], [0,3], [10,3]])
        else:
            rect = np.float32([[0, 0], [10, 0], [10, 17], [0, 17]])
        perspMat = cv2.getPerspectiveTransform(np.float32(angles), rect)

        if self.file_opened:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Save the calculated matrix
        path = os.path.join('parameters', self.cam_name + '-perspective.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, perspMat)

        return perspMat
    
    def _calculate_positions(self, detected_dict):
        """ 
        Calculate positions (in meter) of players, applying fish-eye correction and perspective transformation.
        Given the detected dictionary {ID, bbox}, returns {ID, (bbox, position)}
        """
        if not detected_dict:
            return {}
        
        bboxes = [box for box in detected_dict.values()]
        feet = get_feet_positions(bboxes)
        positions = transform_points(feet, self.K, self.D, self.H)

        PlayerInfo = namedtuple('PlayerInfo', ['bbox', 'position'])

        return {id: PlayerInfo(detected_dict[id], positions[i]) for i, id in enumerate(detected_dict)}
    
    def _calculate_ball_positions(self, balls_bbox):
        """ 
        Calculate positions (in meter) of balls.
        Given the detected ball bounding boxes, return an array of positions.
        """
        if balls_bbox.size == 0:
            return []

        centers = (balls_bbox[:, :2] + balls_bbox[:, 2:]) / 2     # Calculate centers of the bounding boxes
        positions = transform_points(centers, self.K, self.D, self.H)   # Apply fisheye correction and perspective transformation

        return positions

    @staticmethod
    def _save_data_to_csv(data, path):
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for frame_data in data:
                row = [frame_data['frame_num']]
                for player_id, player_info in frame_data['players'].items():
                    row.append(player_id)
                    row.append([int(x) for x in player_info.bbox])
                    row.append(player_info.position)
                writer.writerow(row)
