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
            cv2.putText(frame, f"Player {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    return frame

def draw_ball(frame, ball_list):
    if ball_list is None or not ball_list:
        return frame
    
    for ball in ball_list:
        x1, y1, x2, y2 = ball
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    
    return frame

def draw_mini_court(frame, player_dict = None, balls = None, mouse_pos = None):
    """
    Draw a mini volleyball court on the top left corner of the frame.
    The court is 10x20 meters, with a net at the middle of the court.
    The players are drawn as circles on the court.
    
    Parameters
    ----------
    frame : np.array
        The frame to draw the court on.
    player_dict : dict
        Dictionary containing the players' information.
    balls : list
        List of balls' positions and errors.
    mouse_pos : tuple
        The position of the mouse on the frame.

    Returns
    -------
    frame : np.array
        The frame with the court drawn on it.

    Notes
    -----
    The player_dict should be a dictionary with the following structure:
    {
        0: PlayerInfo,
        1: PlayerInfo,
        ...
    }
    The PlayerInfo object should have (at least) the position attribute in the form of a tuple (x, y).
    The ball_dict should be a list of tuples, each containing the position and error of the ball.
    """

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
    if player_dict is not None:
        for id, player_info in player_dict.items():
            # if id > 5:
            #     id = id % 4
            # cv2.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, players_colors[id], 3, cv2.LINE_AA)
            cv2.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, [0,0,255], 3, cv2.LINE_AA)

    # Draw balls on mini court
    thresh = 2 # above this value, error is considered to much and the ball is drawn in black
    if balls is not None:
        for ball in balls:
            if ball[1] > thresh:
                color = [0, 0, 0]
            else:
                c = int(ball[1]*255/thresh)
                color = [0, c, 255-c]
            # color = [0,255,0] if ball[1] < 0.5 else [0,0,255]
            cv2.circle(frame, (int(ball[0][0]*zoom+field_pos[0]),int(ball[0][1]*zoom+field_pos[1])), 1, color, 3, cv2.LINE_AA)

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

def triangulate_points(O1, O2, points1, points2):
    """
    Compute the 3D positions for each pair of points from two cameras.
    
    Parameters
    ----------
    O1 : array-like (3,)
        First camera coordinates (x, y, z)
    O2 : array-like (3,)
        Second camera coordinates (x, y, z)
    points1 : array-like (N, 2)
        Array of 2D-points from first camera
    points2 : array-like (M, 2)
        Array of 2D-points from second camera
        
    Returns
    -------
    positions3D : np.array (N, M, 3)
        3D positions for all point pairs
    errors : np.array (N, M)
        Errors for all point pairs

    Notes
    -----
    The input points should be in the form (x, y) and represent the 2D coordinates of the tranformed points (in the z=0 plane)
    The cameras inputs should be in the form (x, y, z) 
    The error is calculated as the distance between the two rays connecting the camera origin and its respective 2D point
    """

    if len(points1) == 0 or len(points2) == 0:
        return None, None

    O1 = np.array(O1)
    O2 = np.array(O2)
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    N, M = len(points1), len(points2)
    
    # Create broadcastable arrays (N, M, 2)
    p1 = points1[:, np.newaxis, :]  # (N, 1, 2)
    p2 = points2[np.newaxis, :, :]  # (1, M, 2)
    
    # Compute ray directions for all pairs (N, M, 3)
    d1 = np.empty((N, M, 3))
    d1[..., :2] = p1[..., :2] - O1[:2]
    d1[..., 2] = -O1[2]
    
    d2 = np.empty((N, M, 3))
    d2[..., :2] = p2[..., :2] - O2[:2]
    d2[..., 2] = -O2[2]
    
    # Normalize rays
    d1 /= np.linalg.norm(d1, axis=-1, keepdims=True)
    d2 /= np.linalg.norm(d2, axis=-1, keepdims=True)
    
    # Solve for all pairs using batched least squares
    A = np.stack([d1, -d2], axis=-1)  # (N, M, 3, 2)
    b = O2 - O1  # (3,)
    
    # Manual 2x2 system solving (vectorized)
    # A^T A λ = A^T b
    A_transposed = np.swapaxes(A, -1, -2)  # (N, M, 2, 3)
    AtA = A_transposed @ A  # (N, M, 2, 2)
    Atb = A_transposed @ b  # (N, M, 2)
    
    # Compute determinant
    det = AtA[..., 0, 0] * AtA[..., 1, 1] - AtA[..., 0, 1] * AtA[..., 1, 0]
    inv_det = 1.0 / (det + 1e-10)
    
    # Compute lambdas
    lambda0 = (AtA[..., 1, 1] * Atb[..., 0] - AtA[..., 0, 1] * Atb[..., 1]) * inv_det
    lambda1 = (-AtA[..., 1, 0] * Atb[..., 0] + AtA[..., 0, 0] * Atb[..., 1]) * inv_det
    
    # Compute 3D positions
    P1 = O1 + lambda0[..., np.newaxis] * d1
    P2 = O2 + lambda1[..., np.newaxis] * d2
    positions3D = (P1 + P2) / 2
    errors = np.linalg.norm(P1 - P2, axis=-1)

    # Now we have positions3D and errors in a grid, as (N, M, 3) and (N, M) respectively.
    # We can flatten them to a list of 3D points and errors for each pair of points
    flat_positions3D = positions3D.reshape(-1, 3)
    flat_errors = errors.reshape(-1)

    return flat_positions3D, flat_errors
    
    # We could also convert the 3D points to a list of tuples, instead of a list of lists
    # return [tuple(pos) for pos in flat_positions], flat_errors.tolist()