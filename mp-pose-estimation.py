import cv2
from ultralytics import YOLO
import mediapipe as mp
import time

COMPLEXITY = 2


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=COMPLEXITY, 
    enable_segmentation=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Load YOLOv8 model (using PyTorch)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the smallest and fastest version

# Open video capture
cap = cv2.VideoCapture('input_videos/31-08-2024-10-27.mp4')

def draw_pose(frame, y1, x1, y2, x2, results_pose):
    """
    Function to draw the pose landmarks on the frame.
    """
    if not results_pose.pose_landmarks:
        return frame  # Return the original frame if no pose landmarks found

    mp_drawing.draw_landmarks(
        frame[int(y1):int(y2), int(x1):int(x2)],
        results_pose.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )

    # Place the cropped frame with pose landmarks back into the original frame
    frame[int(y1):int(y2), int(x1):int(x2)] = frame[int(y1):int(y2), int(x1):int(x2)]
    return frame

times = []
border = 10  # Border around the person's bounding box to include more context

cap.set(cv2.CAP_PROP_POS_MSEC, 30000)

while cap.isOpened():

    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv8 for detecting people
    # results = model.track(frame, persist=True)[0]
    results = model(frame)[0]
    id_name_dict = results.names

    # List to store threads (or futures)
    futures = []

    # Extract bounding boxes for detected people
    for box in results.boxes:  # results.pred[0] gives the detections in a frame
        object_cls_id = box.cls.tolist()[0]
        
        if id_name_dict[object_cls_id] != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0]  # xyxy format means x_min y_min , x_max y_max

        # Pose estimation only on lower players
        if y2 < 300:
            continue

        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Apply MediaPipe Pose on the detected person
        cropped_frame = frame[int(y1-border):int(y2+border), int(x1-border):int(x2+border)]
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(cropped_rgb)
        futures.append((results_pose, (x1,y1,x2,y2)))

        # frame = draw_pose(frame, y1, x1, y2, x2, results_pose)

    end_time = time.time()
    frame_time = end_time - start_time
    times.append(frame_time)

    # Display the frame with multi-person pose
    cv2.imshow('Multi-Person Pose Estimation with YOLOv8', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and display the average frame processing time
avg_time = sum(times) / len(times)
print(f"Average frame processing time: {avg_time:.4f} seconds")

cap.release()
cv2.destroyAllWindows()
