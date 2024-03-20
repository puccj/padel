import operator
import os
import cv2
import torch
import torchvision
import numpy as np
from sort import Sort
from utils import get_video_properties, get_dtype

class DetectionModel:
    def __init__(self, dtype=torch.FloatTensor):
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detection_model.type(dtype)  # Also moves model to GPU if available
        self.detection_model.eval()
        self.dtype = dtype
        self.PERSON_LABEL = 1
        self.RACKET_LABEL = 43
        self.BALL_LABEL = 37
        self.PERSON_SCORE_MIN = 0.85
        self.PERSON_SECONDARY_SCORE = 0.3
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0
        self.v_height = 0
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.persons_boxes = {}
        self.persons_dists = {}
        self.persons_first_appearance = {}
        self.counter = 0
        self.num_of_misses = 0
        self.last_frame = None
        self.current_frame = None
        self.next_frame = None
        self.movement_threshold = 200
        self.mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.05)

    def _detect(self, image, person_min_score=None):
        """
        Use deep learning model to detect all person in the image
        """
        if person_min_score is None:
            person_min_score = self.PERSON_SCORE_MIN
        # creating torch.tensor from the image ndarray
        frame_t = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).type(self.dtype)

        # Finding boxes and keypoints
        with torch.no_grad():
            # forward pass
            p = self.detection_model(frame_tensor)

        persons_boxes = []
        probs = []
        for box, label, score in zip(p[0]['boxes'][:], p[0]['labels'], p[0]['scores']):
            if label == self.PERSON_LABEL and score > person_min_score:
                '''cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
                cv2.putText(boxes, 'Person %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)'''
                persons_boxes.append(box.detach().cpu().numpy())
                probs.append(score.detach().cpu().numpy())
        return persons_boxes, probs
    
    def calculate_all_persons_dists(self):
        """
        For each person detected in top half court, calculate the distance their box has moved in the video
        """
        for id, person_boxes in self.persons_boxes.items():
            person_boxes = [box for box in person_boxes if box[0] is not None]
            dist = boxes_dist(person_boxes)
            self.persons_dists[id] = dist
        return self.persons_dists
    
def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2

def boxes_dist(boxes):
    """
    Calculate the cumulative distance of all the boxes
    """
    total_dist = 0
    for box1, box2 in zip(boxes, boxes[1:]):
        box1_center = np.array(center_of_box(box1))
        box2_center = np.array(center_of_box(box2))
        dist = np.linalg.norm(box2_center - box1_center)
        total_dist += dist
    return total_dist


def sections_intersect(sec1, sec2):
    """
    Check if two sections intersect
    """
    if sec1[0] <= sec2[0] <= sec1[1] or sec2[0] <= sec1[0] <= sec2[1]:
        return True
    return False
    
if __name__ == "__main__":
    video = cv2.VideoCapture('../videos/vid.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)
    print(length)
    # Output videos writer
    out = cv2.VideoWriter('output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))
    dtype = get_dtype()
    model = DetectionModel(dtype)
    frame_i = 0
    while True:
        ret, frame = video.read()
        frame_i += 1
        print("processing frame", frame_i)
        if ret:
            # Detect all the persons
            boxes = np.zeros_like(frame)
            persons_boxes, probs = model._detect(frame, model.PERSON_SECONDARY_SCORE)
            if len(persons_boxes) == 0:
                persons_boxes, probs = None, None
                    # Track persons using SORT algorithm
            tracked_objects = model.mot_tracker.update(persons_boxes, probs)
            for det_person in model.persons_boxes.keys():
                model.persons_boxes[det_person].append([None, None, None, None])
            # Mark each person box
            for box in tracked_objects:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
                cv2.putText(frame, f'Player {int(box[4])}', (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                if int(box[4]) in model.persons_boxes.keys():
                    model.persons_boxes[int(box[4])][-1] = box[:4]
                else:
                    model.persons_boxes[int(box[4])] = [box[:4]]
                    model.persons_first_appearance[int(box[4])] = frame_i
                
            out.write(frame)
            #cv2.imshow('df', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows()
        else:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()

    dists = model.calculate_all_persons_dists()
    threshold = 1
    to_del = []
    for key, val in dists.items():
        length = len(model.persons_boxes[key])
        if length > 0:
            average_dist = val / length
        else:
            average_dist = 0
        if average_dist < threshold:
            to_del.append(key)
    for key in to_del:
        dists.pop(key)
        model.persons_first_appearance.pop(key)
    print(dists)
    print(model.persons_first_appearance)
    persons_sections = {key: [val, val + len(model.persons_boxes[key])] for key, val in model.persons_first_appearance.items()}
    print(persons_sections)

    # detections = []

    # while len(dists) > 0:
    #     max_key = max(dists.items(), key=operator.itemgetter(1))[0]
    #     max_sec = persons_sections[max_key].copy()
    #     detections.append(max_key)
    #     persons_sections.pop(max_key)
    #     dists.pop(max_key)
    #     for key, sec in persons_sections.items():
    #         if sections_intersect(max_sec, sec):
    #             if key in dists.keys():
    #                 dists.pop(key)
    # detections = sorted(detections)
    # print(detections)

    # boxes = []
    # for person in detections:
    #     start = model.persons_first_appearance[person]
    #     missing = start - 1 - len(boxes)
    #     boxes.extend([None] * missing)
    #     boxes.extend(model.persons_boxes[person])



    # detections = dists.keys()
    # boxes = {}
    # for person in detections:
    #     boxes[person] = []
    #     start = model.persons_first_appearance[person]
    #     missing = start - 1 - len(boxes[person])
    #     boxes[person].extend([None] * missing)
    #     boxes[person].extend(model.persons_boxes[person])

    # video = cv2.VideoCapture('../videos/vid.mp4')
    # # get videos properties
    # fps, length, v_width, v_height = get_video_properties(video)

    # # Output videos writer
    # out = cv2.VideoWriter('output.avi',
    #                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))


    # frame_i = 0
    # while True:
    #     ret, frame = video.read()
    #     frame_i += 1
    #     if ret:
    #         for box in boxes.values():
    #             if frame_i <= len(box):
    #                 box = box[frame_i - 1]
    #             else:
    #                 box = None
    #             if box is not None and box[0] is not None:
    #                 cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)            
    #             cv2.imshow('df', frame)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 cv2.destroyAllWindows()
    #         out.write(frame)

    #         # if frame_i <= len(boxes):
    #         #     box = boxes[frame_i - 1]
    #         # else:
    #         #     box = None
    #         # if box is not None and box[0] is not None:
    #         #     print(box)
    #         #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
            
    #         # cv2.imshow('df', frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     cv2.destroyAllWindows()
    #         # out.write(frame)
    #     else:
    #         break
    # video.release()
    # out.release()
    # cv2.destroyAllWindows()