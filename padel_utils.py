import os
import numpy as np
import cv2 as cv

def get_foot_position(bbox):
    """Returns the foot position of a player given the bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_feet_positions(bboxes):
    """Returns the foot positions of a list of players given the bounding boxes"""
    return np.float32([get_foot_position(bbox) for bbox in bboxes])

def get_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def transform_points(points, K=None, D=None, H=None):
    """
    Undistort and/or apply perspective transformation to a list of points.
    If K and D are None, only perspective transformation is applied.
    If H is None, only undistortion is applied.
    
    Parameters
    ----------
    points : np.array
        List of points to transform
    K : np.array (3x3)
        Camera fisheye matrix (intrinsic parameters)
    D : np.array (1x5)
        Distortion coefficients (radial and tangential)
    H : np.array (3x3)
        Homography matrix (perspective transformation)
    Returns
    -------
    np.array"""

    # Reshape feet positions to the required shape (n, 1, 2) (the extra 1 required for transformations)
    result = points.reshape(-1, 1, 2)

    if K is not None and D is not None:
        result = cv.fisheye.undistortPoints(result, K, D, None, K)

    if H is not None:
        result = cv.perspectiveTransform(result, H)
        
    return result.reshape(-1,2)    # reshape back to (n,2)

def draw_bboxes(frame, player_dict, show_id = False):
    if player_dict is None or not player_dict:
        return frame
    
    for track_id, player_info in player_dict.items():
        bbox = player_info.bbox
        x1, y1, x2, y2 = bbox
        # bbox[0] = x_min     bbox[1] = y_min
        if show_id:
            cv.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    return frame

def draw_ball(frame, ball_list):
    if ball_list is None or not ball_list:
        return frame
    
    for ball in ball_list:
        x1, y1, x2, y2 = ball
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    
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
    cv.rectangle(shapes, (field_pos[0]-offset, field_pos[1]-offset), (10*zoom+field_pos[0]+offset, 20*zoom+field_pos[1]+offset), bg_color, cv.FILLED)
    cv.rectangle(shapes, field_pos, (10*zoom+field_pos[0], 20*zoom+field_pos[1]), field_color, cv.FILLED)
    out = frame.copy()
    mask = shapes.astype(bool)
    out[mask] = cv.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

    frame = out     # TO SEE: maybe .copy() is needed?

    # Draw court
    cv.line(frame, (field_pos[0]       ,    3  *zoom+field_pos[1]) , (10*zoom+field_pos[0],      3  *zoom+field_pos[1]) , line_color, 2)  #horizontal
    cv.line(frame, (field_pos[0]       ,   17  *zoom+field_pos[1]) , (10*zoom+field_pos[0],     17  *zoom+field_pos[1]) , line_color, 2)  #horizontal
    cv.line(frame, (5*zoom+field_pos[0],int(2.7*zoom+field_pos[1])), ( 5*zoom+field_pos[0], int(17.3*zoom+field_pos[1])), line_color, 2)  #vertical
    cv.line(frame, (field_pos[0]       ,   10  *zoom+field_pos[1]) , (10*zoom+field_pos[0],     10  *zoom+field_pos[1]) , net_color , 1)  #net


    # Draw players on mini court
    if player_dict == None:
        return frame
    
    for id, player_info in player_dict.items():
        # if id > 5:
        #     id = id % 4
        # cv.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, players_colors[id], 3, cv.LINE_AA)
        cv.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, [0,0,255], 3, cv.LINE_AA)

    # Draw mouse position
    if mouse_pos is not None:
        cv.circle(frame, (int(mouse_pos[0]*zoom+field_pos[0]),int(mouse_pos[1]*zoom+field_pos[1])), 1, [0,255,0], 3, cv.LINE_AA)

    return frame