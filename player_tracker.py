from ultralytics import YOLO 
from padel_utils import get_distance

class PlayerTracker:
    def __init__(self, model_path, max_distance_threshold=50):
        self.model = YOLO(model_path)
        self.max_distance_threshold = max_distance_threshold
        self.active_players = {}
        self.inactive_players = {}

    def choose_best_players(self, player_dict):
        """Selects the best 4 players, simply based on their distance to the center (5, 10).
        
        Parameters
        ----------
        player_dict : dict
            A dictionary where keys are track IDs and values are player information objects.
            Each player information object must have a 'position' attribute which is a tuple (x, y).
        
        Returns
        -------
        chosen_players: dict
            A dictionary containing the track IDs and player information of the 4 players closest to the point (5, 10).
        """

        if len(player_dict) <= 4:
            return player_dict
        
        distances = [(track_id, get_distance(player_info.position, (5,10))) for track_id, player_info in player_dict.items()]

        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 4 tracks
        chosen_id = [distances[0][0], distances[1][0], distances[2][0], distances[3][0]]
        # Create a new dictionary with only these 4 and return it
        chosen_players = {track_id: player_info for track_id, player_info in player_dict.items() if track_id in chosen_id}

        return chosen_players

    def detect(self, frame):
        """Detects players in the frame and returns a dictionary with the player ID as key and the bbox as value.
        
            Parameters
            ----------
            frame : numpy.ndarray
                The input frame in which players are to be detected.

            Returns
            -------
            player_dict: dict
                A dictionary where the keys are player IDs and the values are bounding boxes in the format [x_min, y_min, x_max, y_max].

            Notes
            -----
            The method uses a tracking model to persist the tracks across multiple frames.
            Only objects classified as "person" are included in the returned dictionary.
            The ball and other objects are excluded from the results.
        """

        # persist=True tells the tracks that this is not just an individaul frame, but other
        # frames will be given afterwards and the model should persist the track in those frames.
        results = self.model.track(frame, persist=True)[0]
        # results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
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

        return player_dict    