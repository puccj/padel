from ultralytics import YOLO 
import cv2 as cv
from padel_utils import get_distance, get_centroid

class PlayerTracker:
    def __init__(self, model_path, max_distance_threshold=50):
        self.model = YOLO(model_path)
        self.max_distance_threshold = max_distance_threshold
        self.active_players = {}
        self.inactive_players = {}

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]    # Choose the player according only to the first frame
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, player_dict):
        if len(player_dict) <= 4:
            return player_dict
        
        distances = [(track_id, get_distance(player_info.position, (5,10))) for track_id, player_info in player_dict.items()]

        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 4 tracks
        chosen_id = [distances[0][0], distances[1][0], distances[2][0], distances[3][0]]
        # Create a new dictionary with only these 4 and return it
        chosen_players = {track_id: player_info for track_id, player_info in player_dict.items() if track_id in chosen_id}

        # TODO: instead (at the end) take the most present players throwout the video 
        return chosen_players

    def detect(self, frame):
        # persist=True tells the tracks that this is not just an individaul frame, but other
        # frames will be given afterwards and the model should persist the track in those frames.
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        # Here we only want people -> we'll exclude everything else. 
        # Also, since we have another separate track for the ball, we exclude that as well.
        for i, box in enumerate(results.boxes):
            track_id = box.id
            if track_id is None:
                track_id = i
            else:
                track_id = int(track_id.tolist()[0])
            result = box.xyxy.tolist()[0]       #xyxy format means x_min y_min , x_max y_max
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        # Reassign IDs
        self.reassign_ids(player_dict)

        return player_dict

    def reassign_ids(self, detected_dict):
        new_active_players = {}
        new_inactive_players = {}

        for new_id, bbox in detected_dict.items():
            new_position = get_centroid(bbox)
            reassigned = False

            # Check if this new player can be an inactive player re-entering the scene
            for old_id, old_info in self.inactive_players.items():
                old_position = old_info['position']
                if get_distance(new_position, old_position) < self.max_distance_threshold:
                    new_active_players[old_id] = {'bbox': bbox, 'position': new_position}
                    reassigned = True
                    break

            if not reassigned:
                new_active_players[new_id] = {'bbox': bbox, 'position': new_position}

        self.active_players = new_active_players
        self.inactive_players = new_inactive_players

    def draw_bboxes(self, frame, player_dict):
        if player_dict is None:
            return frame
        
        for track_id, player_info in player_dict.items():
            bbox = player_info.bbox
            x1, y1, x2, y2 = bbox
            # bbox[0] = x_min     bbox[1] = y_min
            cv.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        return frame


    