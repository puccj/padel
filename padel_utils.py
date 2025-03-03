import numpy as np
import cv2

def get_foot_position(bbox):
    """Returns the foot position of a player given the bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_feet_positions(bboxes):
    """Returns the foot positions of a list of players given the bounding boxes"""
    return np.float32([get_foot_position(bbox) for bbox in bboxes])

def get_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def load_fisheye_params(path):
    """
    Load fisheye parameters from a file and return the camera intrinsic matrix (K) and distortion coefficients (D).
    If the file is not found, fisheye parameters are calculated using gui_calib.py.

    Parameters
    ----------
    path : str
        Path to the file containing the fisheye parameters.

    Returns
    -------
    K : np.array (3x3)
        Camera matrix (intrinsic parameters)
    D : np.array (1x4)
        Distortion coefficients (radial and tangential)

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If no parameters are found in the file.
    """
    parameters = {}
    
    with open(path, 'r') as file:
        for line in file:
            # Split the line by '=' to separate key and value
            key, value = map(str.strip, line.strip().split('='))
            parameters[key] = float(value)  # Convert value to float

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

    mtx = np.array([[fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]])
    dist = np.array([[k1, k2, p1, p2]])

    return mtx, dist


def transform_points(points, K=None, D=None, H=None):
    """
    Undistort and/or apply perspective transformation to a list of points.
    If K and D are None, only perspective transformation is applied.
    If H is None, only undistortion is applied.
    
    Parameters
    ----------
    points : np.array (nx2) or list of (x,y) tuples
        List of points to transform. If a 
    K : np.array (3x3)
        Camera fisheye matrix (intrinsic parameters)
    D : np.array (1x4)
        Distortion coefficients (radial and tangential)
    H : np.array (3x3)
        Homography matrix (perspective transformation)
    
    Returns
    -------
    transformed_points : np.array
        List of transformed points
    """

    if points is None or len(points) == 0:
        return np.array([])

    # Convert list of tuples to numpy array
    if isinstance(points, list):
        points = np.float32(points)

    # Reshape feet positions to the required shape (n, 1, 2) (the extra 1 required for transformations)
    result = points.reshape(-1, 1, 2)

    if K is not None and D is not None:
        result = cv2.fisheye.undistortPoints(result, K, D, None, K)

    if H is not None:
        result = cv2.perspectiveTransform(result, H)
        
    return result.reshape(-1,2)    # reshape back to (n,2)

def draw_bboxes(frame, player_dict, show_id = False):
    if player_dict is None or not player_dict:
        return frame
    
    for track_id, player_info in player_dict.items():
        bbox = player_info.bbox
        x1, y1, x2, y2 = bbox
        # bbox[0] = x_min     bbox[1] = y_min
        if show_id:
            cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    return frame

def draw_ball(frame, ball_list):
    if ball_list is None or not ball_list:
        return frame
    
    for ball in ball_list:
        x1, y1, x2, y2 = ball
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    
    return frame

def draw_mini_court(frame, player_dict = None, mouse_pos = None):
    # Variables
    #zoom = 25
    zoom = int(frame.shape[0] / 40)  #height of frame/40
    offset = 2*zoom
    bg_color = (209, 186, 138) #(255, 255, 255)
    field_color = (129, 94, 61)
    line_color = (255,255,255)
    net_color = (0,0,0)
    alpha = 0.2
    field_pos = (offset*2, offset*2)
    # players_colors = [(0,0,0), (0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,255,255)]

    # Draw rectangles
    shapes = np.zeros_like(frame,np.uint8)
    cv2.rectangle(shapes, (field_pos[0]-offset, field_pos[1]-offset), (10*zoom+field_pos[0]+offset, 20*zoom+field_pos[1]+offset), bg_color, cv2.FILLED)
    cv2.rectangle(shapes, field_pos, (10*zoom+field_pos[0], 20*zoom+field_pos[1]), field_color, cv2.FILLED)
    out = frame.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

    frame = out     # TO SEE: maybe .copy() is needed?

    # Draw court
    cv2.line(frame, (field_pos[0]       ,    3  *zoom+field_pos[1]) , (10*zoom+field_pos[0],      3  *zoom+field_pos[1]) , line_color, 2)  #horizontal
    cv2.line(frame, (field_pos[0]       ,   17  *zoom+field_pos[1]) , (10*zoom+field_pos[0],     17  *zoom+field_pos[1]) , line_color, 2)  #horizontal
    cv2.line(frame, (5*zoom+field_pos[0],int(2.7*zoom+field_pos[1])), ( 5*zoom+field_pos[0], int(17.3*zoom+field_pos[1])), line_color, 2)  #vertical
    cv2.line(frame, (field_pos[0]       ,   10  *zoom+field_pos[1]) , (10*zoom+field_pos[0],     10  *zoom+field_pos[1]) , net_color , 1)  #net


    # Draw players on mini court
    if player_dict == None:
        return frame
    
    for id, player_info in player_dict.items():
        # if id > 5:
        #     id = id % 4
        # cv2.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, players_colors[id], 3, cv2.LINE_AA)
        cv2.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, [0,0,255], 3, cv2.LINE_AA)

    # Draw mouse position
    if mouse_pos is not None:
        cv2.circle(frame, (int(mouse_pos[0]*zoom+field_pos[0]),int(mouse_pos[1]*zoom+field_pos[1])), 1, [0,255,0], 3, cv2.LINE_AA)

    return frame

def triangulate_point(O1, O2, point1, point2):
    """
    Triangulates the 3D position of a point given its 2D projections from two different cameras. Uses least-squares

    Parameters
    ----------
    O1 : array-like
        The origin (position) of the first camera in 3D space.
    O2 : array-like
        The origin (position) of the second camera in 3D space.
    point1 : array-like
        The 2D coordinates of the point in the image plane of the first camera.
    point2 : array-like
        The 2D coordinates of the point in the image plane of the second camera.

    Returns
    -------
    position3D : np.array
        The 3D coordinates of the triangulated point.
    error : float
        The error of the triangulation (distance between the two rays).
    """

    # Rays from the camera origins to the points
    d1 = np.array([point1[0] - O1[0], point1[1] - O1[1], -O1[2]])
    d2 = np.array([point2[0] - O2[0], point2[1] - O2[1], -O2[2]]) 

    # Normalize the rays TODO: this is not necessary, check if faster without
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)

    # Coefficients of the system of equations
    A = np.vstack([d1, -d2]).T
    b = O2 - O1

    # Apply least-squares to solve the system of equations
    lambdas = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the 3D position of the point
    P1 = O1 + lambdas[0] * d1
    P2 = O2 + lambdas[1] * d2
    P3D = (P1 + P2) / 2
    error = np.linalg.norm(P1 - P2)
    
    return P3D, error

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