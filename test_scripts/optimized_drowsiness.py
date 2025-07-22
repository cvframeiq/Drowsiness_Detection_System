import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

#FaceMEsh with lighter config
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

#EYE Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.2
DROWSY_FRAMES = 20
ear_history = deque(maxlen=5)
closed_eye_counter=0

cap = cv2.VideoCapture(0)
frame_counter = 0
start_time = time.time()

def calculate_EAR(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C)
    return ear, points

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #Resizing for performance improvement
    frame = cv2.resize(frame, (640, 480))

    frame_counter += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w = frame.shape[:2]

    status = "No face detected"
    color = (200, 200, 200)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        left_ear, left_points = calculate_EAR(face_landmarks.landmark, LEFT_EYE, w, h)
        right_ear, right_points = calculate_EAR(face_landmarks.landmark, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_history.append(avg_ear)
        smoothed_ear = sum(ear_history) / len(ear_history)

        #State of eyes
        if smoothed_ear < EAR_THRESHOLD:
            closed_eye_counter += 1
            if closed_eye_counter >= DROWSY_FRAMES:
                status = "Drowsiness detected!"
                color = (0, 0, 255)
            else:
                status = "Eyes closed"
                color = (0, 255, 255)
        else:
            closed_eye_counter = 0
            status = "Eyes Open"
            color = (0, 255, 0)

        for pt in left_points + right_points:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        cv2.putText(frame, f"EAR: {smoothed_ear: .2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        

    #show Status
    cv2.putText(frame, f"Status: {status}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    #FPS for every 10 frames
    if frame_counter % 10 == 0:
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        start_time = end_time
        cv2.putText(frame, f"FPS: {int(fps)}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
    cv2.imshow("Optimized Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
