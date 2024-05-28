import os
import numpy as np
import cv2 as cv

def ensure_directory_exists(file_path):
    # Extract the directory from the given file path
    directory = os.path.dirname(file_path)
    
    # If directory is empty, don't attempt to create it
    if not directory:
        return
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_feet_positions(bboxes):
    return np.float32([get_foot_position(bbox) for bbox in bboxes])

def get_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_mini_court(frame, player_dict = None):
    # Variables
    #zoom = 25
    zoom = int(frame.shape[0] / 40)  #hight of frame/40
    offset = 2*zoom
    bg_color = (209, 186, 138) #(255, 255, 255)
    field_color = (129, 94, 61)
    line_color = (255,255,255)
    net_color = (0,0,0)
    alpha = 0.2
    field_pos = (offset*2, offset*2)
    players_color = [(0,0,0), (0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,255,255)]

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
        if id > 5:
            id = id % 4
        cv.circle(frame, (int(player_info.position[0]*zoom+field_pos[0]),int(player_info.position[1]*zoom+field_pos[1])), 1, players_color[id], 3, cv.LINE_AA)

    return frame


def draw_stats(frame, frame_data):    
    zoom = int(frame.shape[1] / 71)
    offset = 2*zoom
    box_pos = (50*zoom, offset)
    
    overlay = frame.copy()
    cv.rectangle(overlay, box_pos, (20*zoom+box_pos[0], 5*zoom+box_pos[1]), (0, 0, 0), -1) #-1 = filled
    alpha = 0.5 
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    text = "     Player 1   Player 2   Player 3   Player 4"
    frame = cv.putText(frame, text, (box_pos[0]+ 3*zoom, box_pos[1]+1*zoom), cv.FONT_HERSHEY_SIMPLEX, 0.0222*zoom, (255, 255, 255), 2)
    
    text = "Kilometers"
    frame = cv.putText(frame, text, (box_pos[0]+2, box_pos[1]+3*zoom), cv.FONT_HERSHEY_SIMPLEX, 0.01667*zoom, (255, 255, 255), 1)
    text = f"{frame_data['player_1_distance']:.1f} m      {frame_data['player_2_distance']:.1f} m      "
    text += f"{frame_data['player_3_distance']:.1f} m      {frame_data['player_4_distance']:.1f} m"
    frame = cv.putText(frame, text, (box_pos[0]+5*zoom, box_pos[1]+3*zoom), cv.FONT_HERSHEY_SIMPLEX, 0.0185*zoom, (255, 255, 255), 2)

    text = "Player Speed"
    frame = cv.putText(frame, text, (box_pos[0]+2, box_pos[1]+4*zoom), cv.FONT_HERSHEY_SIMPLEX, 0.01667*zoom, (255, 255, 255), 1)
    text = f"{frame_data['player_1_speed']:.1f} km/h   {frame_data['player_2_speed']:.1f} km/h   "
    text += f"{frame_data['player_3_speed']:.1f} km/h   {frame_data['player_4_speed']:.1f} km/h"
    frame = cv.putText(frame, text, (box_pos[0]+5*zoom, box_pos[1]+4*zoom), cv.FONT_HERSHEY_SIMPLEX, 0.0185*zoom, (255, 255, 255), 2)

    return frame
