import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math
from collections import deque
import winsound

# Loading trained model
model = tf.keras.models.load_model('eye_state_classifier.keras')

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
YAWN_THRESHOLD = 0.6
eye_closed_counter = 0

# Eye landmarks
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Beep alert
def play_alert():
    winsound.Beep(1000, 700)

# Distance calc
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# EAR
def calculate_ear(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Webcam 
# You can change the index value here if it doesn't work on 0.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    label = "No Face"
    color = (200, 200, 200)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # EAR
        left_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_IDX]
        right_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_IDX]
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

        # Face crop for model
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        face_crop = frame[y_min:y_max, x_min:x_max]

        if face_crop.size != 0:
            try:
                face_resized = cv2.resize(face_crop, (224, 224))
                face_input = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_input, axis=0)
                model_prediction = model.predict(face_input, verbose=0)[0][0]
            except:
                model_prediction = 1.0
        else:
            model_prediction = 1.0

        # Yawning detection
        top_lip = (landmarks[13].x * w, landmarks[13].y * h)
        bottom_lip = (landmarks[14].x * w, landmarks[14].y * h)
        left_mouth = (landmarks[78].x * w, landmarks[78].y * h)
        right_mouth = (landmarks[308].x * w, landmarks[308].y * h)

        vertical_dist = euclidean(top_lip, bottom_lip)
        horizontal_dist = euclidean(left_mouth, right_mouth)
        yawn_ratio = vertical_dist / horizontal_dist

        print(f"EAR: {ear:.3f} | Model: {model_prediction:.2f} | Yawn Ratio: {yawn_ratio:.2f} | EyeCounter: {eye_closed_counter}")

        if yawn_ratio > YAWN_THRESHOLD:
            label = "Yawning"
            color = (255, 0, 0)
            play_alert()
        elif ear < EAR_THRESHOLD and model_prediction < 0.5:
            eye_closed_counter += 1
            if eye_closed_counter >= CONSEC_FRAMES:
                label = "Drowsy"
                color = (0, 0, 255)
                play_alert()
        else:
            eye_closed_counter = 0
            label = "Awake"
            color = (0, 255, 0)

    # Display
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.imshow("Hybrid Drowsiness + Yawn Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
