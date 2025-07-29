from ultralytics import YOLO

class BallTracker:
    def __init__(self,model_path, conf_thres=0.15):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        """ Detects the ball in the frame and returns a list of tuples containing bboxes and confidence scores."""
        results = self.model(frame, conf=self.conf_thres, verbose=False)[0]
        
        balls = [(box.xyxy.tolist()[0], box.conf.item()) for box in results.boxes] #if box.conf.item() > self.conf_thres)

        # Equivalent to:
        # balls = []
        # for box in results.boxes:
        #     bbox = box.xyxy.tolist()[0]
        #     conf = box.conf.item()
        #     balls.append((bbox, conf))

        return balls
