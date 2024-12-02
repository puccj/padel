import cv2
from ultralytics import YOLO
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load YOLOv8 model (using PyTorch)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the smallest and fastest version

# Open video capture
cap = cv2.VideoCapture('input_videos/31-08-2024-10-27.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv8 for detecting people
    # results = model.track(frame, persist=True)[0]
    results = model(frame)[0]
    id_name_dict = results.names

    # Extract bounding boxes for detected people
    for box in results.boxes:  # results.pred[0] gives the detections in a frame
        object_cls_id = box.cls.tolist()[0]
        
        if id_name_dict[object_cls_id] != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0]  # xyxy format means x_min y_min , x_max y_max
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Apply MediaPipe Pose on the detected person
        cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(cropped_rgb)

        if not results_pose.pose_landmarks:
            continue

        mp_drawing.draw_landmarks(
            cropped_frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        # Place the cropped frame with poses back into the original frame
        frame[int(y1):int(y2), int(x1):int(x2)] = cropped_frame

    # Display the frame with multi-person pose
    cv2.imshow('Multi-Person Pose Estimation with YOLOv8', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
