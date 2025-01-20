from ultralytics import YOLO

class BallTracker:
    def __init__(self,model_path, conf_thres=0.15):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        """ Detects the ball in the frame and returns a dictionary with the ball ID as key and the bbox as value """

        # persist=True tells the tracks that this is not just an individaul frame, but other
        # frames will be given afterwards and the model should persist the track in those frames.
        results = self.model(frame, conf=self.conf_thres)[0]
        id_name_dict = results.names

        balls = []
        # Here we only want the ball -> we'll exclude everything else. 
        # Also, since we have another separate track for the players, we exclude them as well.
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            balls.append(result)

        return balls
