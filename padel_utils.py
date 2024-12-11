import os
import numpy as np
import cv2 as cv

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_feet_positions(bboxes):
    return np.float32([get_foot_position(bbox) for bbox in bboxes])

def get_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

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

def draw_mini_court(frame, player_dict = None):
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

    return frame