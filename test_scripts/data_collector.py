import cv2
import mediapipe as mp
import os
import time
from datetime import datetime
import numpy as np

#Config
EAR_THRESHOLD = 0.25
SAVE_INTERVAL = 1  # seconds between saves
DATASET_PATH = "dataset"
LABELS = ["open", "closed"]


# Setup Folders 
for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)
    os.makedirs(folder, exist_ok=True)

# Mediapipe Setup 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Eye landmark indices from Mediapipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# EAR Calculation 
def calculate_ear(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Video Capture 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not access webcam.")
    exit()

print("Data Collector Started")
last_saved = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        left_ear = calculate_ear(coords, LEFT_EYE)
        right_ear = calculate_ear(coords, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2

        status = "closed" if avg_ear < EAR_THRESHOLD else "open"

        # Draw EAR and status on frame
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {status.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save image every SAVE_INTERVAL seconds
        if time.time() - last_saved > SAVE_INTERVAL:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{status}_{timestamp}.jpg"
            save_path = os.path.join(DATASET_PATH, status, filename)
            cv2.imwrite(save_path, frame)
            print(f"Saved {status} frame to {save_path}")
            last_saved = time.time()

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
