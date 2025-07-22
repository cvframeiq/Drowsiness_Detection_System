import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
import math
import time
from collections import deque
import winsound
import threading
import datetime

# Streamlit UI
st.set_page_config(page_title="Drowsiness Detector", layout="wide")
st.title("Drowsiness Detector")
FRAME_WINDOW = st.image([])

# Session state flags
if 'detecting' not in st.session_state:
    st.session_state.detecting = False
if 'eye_closed_counter' not in st.session_state:
    st.session_state.eye_closed_counter = 0
if 'nod_history' not in st.session_state:
    st.session_state.nod_history = deque(maxlen=10)
if 'prev_nose_y' not in st.session_state:
    st.session_state.prev_nose_y = None

# Beep limit (2sec for my convinence)
if 'beep_count' not in st.session_state:
    st.session_state.beep_count = {'drowsy': 0, 'yawn': 0, 'nod': 0}

# Model & MediaPipe setup
model = tf.keras.models.load_model('eye_state_classifier.h5')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Constants
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 15
NOD_THRESHOLD = 15
NOD_HISTORY = 10
YAWN_THRESHOLD = 0.6

# Utility functions
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def log_event(message):
    with open("drowsiness_log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")

def play_alert():
    threading.Thread(target=lambda: winsound.Beep(1000, 700), daemon=True).start()

# Start and Stop buttons
col1, col2 = st.columns(2)
if col1.button("Start Drowsiness Detection"):
    st.session_state.detecting = True
if col2.button("Stop Detection"):
    st.session_state.detecting = False

# Main detection loop (Drowsiness detection is done by EAR & Model both for better accuracy)
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)

    while st.session_state.detecting and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        label = ""
        color = (255, 255, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size != 0:
                try:
                    resized_face = cv2.resize(face_crop, (224, 224))
                    input_face = resized_face.astype("float32") / 255.0
                    input_face = np.expand_dims(input_face, axis=0)
                    prediction = model.predict(input_face, verbose=0)[0][0]
                except:
                    prediction = 1
            else:
                prediction = 1

            left_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_IDX]
            right_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_IDX]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f} | Model: {prediction:.2f}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            # Drowsiness detection
            if ear < EAR_THRESHOLD and prediction < 0.5:
                st.session_state.eye_closed_counter += 1
                if st.session_state.eye_closed_counter >= CONSEC_FRAMES:
                    label = "Drowsy! Wake up!"
                    color = (0, 0, 255)
                    log_event("Drowsiness detected")
                    if st.session_state.beep_count['drowsy'] < 2:
                        play_alert()
                        st.session_state.beep_count['drowsy'] += 1
            else:
                st.session_state.eye_closed_counter = 0
                label = "Attentive"
                color = (0, 255, 0)
                st.session_state.beep_count['drowsy'] = 0  # Reset beep count when user is attentive

            # Yawning detection
            top_lip = (landmarks[13].x * w, landmarks[13].y * h)
            bottom_lip = (landmarks[14].x * w, landmarks[14].y * h)
            left_corner = (landmarks[78].x * w, landmarks[78].y * h)
            right_corner = (landmarks[308].x * w, landmarks[308].y * h)
            vertical = euclidean(top_lip, bottom_lip)
            horizontal = euclidean(left_corner, right_corner)
            yawn_ratio = vertical / horizontal

            if yawn_ratio > YAWN_THRESHOLD:
                label = "Yawning"
                color = (255, 0, 0)
                log_event("Yawning detected")
                if st.session_state.beep_count['yawn'] < 2:
                    play_alert()
                    st.session_state.beep_count['yawn'] += 1
            else:
                st.session_state.beep_count['yawn'] = 0  # Reset when mouth closes


            # Head nodding detection
            nose_y = landmarks[1].y * h
            if st.session_state.prev_nose_y is not None:
                diff = st.session_state.prev_nose_y - nose_y
                st.session_state.nod_history.append(diff)
                if len(st.session_state.nod_history) == NOD_HISTORY and sum(st.session_state.nod_history) > NOD_THRESHOLD:
                    label = "Head Nodding"
                    color = (0, 255, 255)
                    st.session_state.nod_history.clear()
                    log_event("Head Nodding detected")
                    if st.session_state.beep_count['nod'] < 2:
                        play_alert()
                        st.session_state.beep_count['nod'] += 1
                else:
                    st.session_state.beep_count['nod'] = 0


        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
