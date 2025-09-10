"""Debugging and testing utilities, that won't be used during normal execution."""

import cv2
from padel_utils import transform_points
from ultralytics import YOLO
from itertools import chain
import numpy as np
import csv
import os

point = None

def click_event(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print(f"Selected point: {x}, {y}")

def select_point(image_path):
    """
    Open image and select a point with mouse click.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
        
    Returns
    -------
    point : tuple
        Coordinates of the selected point.

    Raises
    ------
    FileNotFoundError
        If the image is not found.
    """
    global point
    point = None    # reset point

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Image not found")

    cv2.imshow("Click a point and press any key", img)
    cv2.setMouseCallback("Click a point and press any key", click_event)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if point is None:
        print("No point selected!")
        return None

    return point

def test_transform(image_path, K,D,H):
    """
    Open image, let the user click a point and print the transformed coordinates.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    K : numpy.ndarray
        Camera matrix.
    D : numpy.ndarray
        Distortion coefficients.
    H : numpy.ndarray
        Homography matrix.

    Raises
    ------
    FileNotFoundError
        If the image is not found.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")
    

    while True:
        global point
        # point = None
        cv2.imshow("Click a point to calculate its coordinates. Press q to close", img)
        cv2.setMouseCallback("Click a point to calculate its coordinates. Press q to close", click_event)
        
        if point is not None:
            transformed = transform_points([point], K, D, H)[0]
            print("Original point:", point)
            print("Transformed point:", transformed)
            print()
            point = None

        if cv2.waitKey(10) == ord("q"):
            cv2.destroyAllWindows()
            break

def detect_ball(video_path, K, D, H, output_path, 
                threshold=30, 
                min_area=3, 
                kernel_size=20, 
                skip_frame=0, 
                sync_frame=0, 
                model="models/yolov8x-seg.pt",
                show=False,
                delay=1):
    """
    Detect the ball in a video and save the output data.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    K : numpy.ndarray
        Camera matrix.
    D : numpy.ndarray
        Distortion coefficients.
    H : numpy.ndarray
        Homography matrix.
    output_path : str
        Path to the output data file.
    threshold : int, optional
        Threshold for the ball detection. The default is 30.
    min_area : int, optional
        Minimum area of the ball. The default is 3.
    kernel_size : int, optional
        Size of the kernel for dilation. The default is 20.
    skip_frame : int, optional
        Number of frames to skip at the beginning to synchronize the two cameras. The default is 0.
    sync_frame : int, optional
        Number of frames to skip every iteration to synchronize the two cameras.
        e.g., put 1 to discard half of the frames for 20fps vs 10fps videos. The default is 0.
    model : str, optional
        Path to the YOLO model. The default is "model/yolo8x-seg.pt".
    show : bool, optional
        Show the video while processing. The default is False.
    delay : int, optional
        Delay between frames in milliseconds. The default is 1.

    Returns
    -------
    output_path : str
        Path to the output data file.

    Raises
    ------
    FileNotFoundError
        If the video file is not found.
    """

    model = YOLO(model)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Video not found")
    
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_results = model(old_frame)[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, sync_frame)

    for _ in range(skip_frame):
        cap.read()

    # Create the file (erasing it if it already exists)
    with open(output_path, 'w', newline=''):
        pass

    frame_num = 0

    while True:
        for _ in range(sync_frame): # Skip frames
            cap.read()

        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(old_gray, frame_gray)
        _, frame_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        results = model(frame)[0]

        # Kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if results.masks is not None:
            for mask, cls in zip(results.masks.data, results.boxes.cls):
                if int(cls) != 0 and int(cls) != 38:
                    continue

                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255      # Binarize
                mask = cv2.dilate(mask, kernel, iterations=1)   # Dilate
                frame_diff[mask > 0] = 0    # Remove players and rackets

        if old_results.masks is not None:
            for mask, cls in zip(old_results.masks.data, old_results.boxes.cls):
                if int(cls) != 0 and int(cls) != 38:    # 0 = 'person', 38 = 'tennis racket'
                    continue

                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255      # Binarize
                mask = cv2.dilate(mask, kernel, iterations=1)   # Dilate
                frame_diff[mask > 0] = 0    # Remove players and rackets
                if show:
                    frame[mask > 0] = 0

        balls = []  # Ball position in camera frame (px)

        # Find contours
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                balls.append(np.float32([x+w/2, y+h/2]))
                if show:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        balls_position = transform_points(balls, K, D, H)   # in world frame (m)

        frame_data = {
            "frame": frame_num,
            "balls": balls_position
        }

        with open(output_path, "a", newline='') as f:
            writer = csv.writer(f)
            row = [frame_data["frame"]] + [pos for pos in frame_data["balls"]]
            writer.writerow(row)

        # Show the video
        if show:
            cv2.imshow("Frame", frame)
            cv2.imshow("Diff", frame_diff)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        old_gray = frame_gray
        old_results = results
        frame_num += 1

    cap.release()
    return output_path
    
def load_data(path):
    """Load (ball) data from csv file
    Parameters
    ----------
    path : str
        Path to the csv file
    
    Returns
    -------
    data : list of lists of tuples
        Each element of the list represents a frame. Each frame is a list of tuples, each tuple is a 2D point

    Raises
    ------
    FileNotFoundError
        If the file is not found

    Notes
    -----
    1) The csv file should have the following format: frame_number, [x1,y1], [x2,y2], ...
    2) If the frame number are irregular (not starting from 0 or not consecutive), 
    maybe it's better to return a dictionary with the frame number as key: data[frame_num] = detections
    """
    
    data = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    if os.path.getsize(path) == 0:
        return data

    with open(path, mode='r') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            detections = row[1:]
            frame_data = []
            for detection in detections:
                pos_str = detection.strip('[] ').split()
                pos = tuple(float(coord) for coord in pos_str)
                frame_data.append(pos)
            
            data.append(frame_data)
    
    return data

def generate_test_data(num_points=1000):
    """Generate random test data with realistic camera geometry"""
    np.random.seed(42)
    
    # Camera positions (realistic stereo setup)
    O1 = np.array([-0.5, 0, 2.0])  # Left camera
    O2 = np.array([0.5, 0, 2.0])   # Right camera
    
    # Generate random 3D points in front of cameras
    points3d = np.random.randn(num_points, 3) * 5
    points3d[:, 2] = np.abs(points3d[:, 2]) + 3  # Ensure positive Z
    
    # Project to 2D (simple perspective projection)
    points1 = points3d[:, :2] / points3d[:, 2:] + np.random.normal(0, 0.01, (num_points, 2))
    points2 = (points3d[:, :2] - (O2[:2] - O1[:2])) / points3d[:, 2:] + np.random.normal(0, 0.01, (num_points, 2))
    
    return O1, O2, points1, points2, points3d

if __name__ == "__main__":
    point1 = select_point("input_videos/primo test pallina/cam1.png")
    point2 = select_point("input_videos/primo test pallina/cam2.png")
    print("Punto 1:", point1)
    print("Punto 2:", point2)

def triangulate_points(O1, O2, points1_array, points2_array):
    """
    Vectorized version of 3D-positions triangulation from 2D projections.

    Parameters
    ----------
    O1 : array-like, shape (3,)
        Origin (position) of the first camera.
    O2 : array-like, shape (3,)
        Origin (position) of the second camera.
    points1 : array-like, shape (N, 2)
        2D coordinates in the first camera's image plane.
    points2 : array-like, shape (N, 2)
        2D coordinates in the second camera's image plane.

    Returns
    -------
    positions3D : np.array, shape (N, 3)
        3D coordinates of the triangulated points.
    errors : np.array, shape (N,)
        Reprojection errors (distance between rays).

    Notes
    -----
    AI-generated. Vectorized version of the triangulate_point function.
    """

    O1 = np.array(O1)
    O2 = np.array(O2)
    points1 = np.array(points1_array)
    points2 = np.array(points2_array)
    N = points1.shape[0]

    # Compute ray directions for all points
    d1 = np.column_stack([
        points1[:, 0] - O1[0],
        points1[:, 1] - O1[1],
        -np.full(N, O1[2])
    ])
    d2 = np.column_stack([
        points2[:, 0] - O2[0],
        points2[:, 1] - O2[1],
        -np.full(N, O2[2])
    ])

    # Normalize rays (avoid division by zero)
    norm1 = np.linalg.norm(d1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(d2, axis=1, keepdims=True)
    d1 = np.divide(d1, norm1, where=norm1 != 0)
    d2 = np.divide(d2, norm2, where=norm2 != 0)

    # Batch least-squares setup
    A = np.stack([d1, -d2], axis=2)  # Shape (N, 3, 2)
    b = O2 - O1  # Shape (3,)

    # Solve (A^T A)λ = A^T b for all points
    A_transposed = np.transpose(A, (0, 2, 1))  # Shape (N, 2, 3)
    AtA = A_transposed @ A  # Shape (N, 2, 2)
    Atb = A_transposed @ b  # Shape (N, 2)

    # Explicit 2x2 matrix inversion (vectorized)
    a, b_ = AtA[:, 0, 0], AtA[:, 0, 1]
    c, d = AtA[:, 1, 0], AtA[:, 1, 1]
    det = a * d - b_ * c
    inv_det = np.divide(1.0, det, where=det != 0)

    # Compute inverse(AtA) * Atb
    lambda0 = (d * Atb[:, 0] - b_ * Atb[:, 1]) * inv_det
    lambda1 = (-c * Atb[:, 0] + a * Atb[:, 1]) * inv_det
    lambdas = np.column_stack([lambda0, lambda1])

    # Calculate 3D positions and errors
    P1 = O1 + lambdas[:, [0]] * d1
    P2 = O2 + lambdas[:, [1]] * d2
    positions3D = (P1 + P2) / 2
    errors = np.linalg.norm(P1 - P2, axis=1)

    return positions3D, errors