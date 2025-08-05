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

# ==================== UI CONFIG ====================
st.set_page_config(page_title="🛡️ Drowsiness Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>🛌 Real-Time Drowsiness Detection</h1>", unsafe_allow_html=True)

video_col, status_col = st.columns([4, 1])
FRAME_WINDOW = video_col.empty()

# Sidebar
st.sidebar.header("📋 About")
st.sidebar.info("Detects drowsiness, yawning, and head nodding in real time.\nPowered by MediaPipe, TensorFlow, OpenCV.")

# ==================== STATE ====================
if 'detecting' not in st.session_state:
    st.session_state.detecting = False
    st.session_state.eye_closed_counter = 0
    st.session_state.nod_history = deque(maxlen=10)
    st.session_state.prev_nose_y = None
    st.session_state.beep_count = {'drowsy': 0, 'yawn': 0, 'nod': 0}

# ==================== MODEL + MEDIAPIPE ====================
model = tf.keras.models.load_model('../model_training/eye_state_classifier.keras')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==================== CONSTANTS ====================
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 8
YAWN_THRESHOLD = 0.6
NOD_THRESHOLD = 15
NOD_HISTORY = 10

# ==================== UTILS ====================
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def play_alert(count=1, frequency=1000, duration=600):
    def beep_seq():
        for _ in range(count):
            winsound.Beep(frequency, duration)
            time.sleep(0.1)
    threading.Thread(target=beep_seq, daemon=True).start()

def log_event(event):
    now = datetime.datetime.now()
    line = f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {event}\n"
    with open("drowsiness_log.txt", "a", encoding="utf-8") as f:
        f.write(line)

# ==================== BUTTONS ====================
start, stop = st.columns(2)
if start.button("▶️ Start Detection"):
    st.session_state.detecting = True
if stop.button("⏹️ Stop Detection"):
    st.session_state.detecting = False

# ==================== DETECTION LOOP ====================
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    time.sleep(1)

    while st.session_state.detecting and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam error.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        label = "🟢 Attentive"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_eye = [(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE_IDX]
            right_eye = [(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE_IDX]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            x_min = int(min([p.x for p in lm]) * w)
            x_max = int(max([p.x for p in lm]) * w)
            y_min = int(min([p.y for p in lm]) * h)
            y_max = int(max([p.y for p in lm]) * h)
            face_crop = frame[y_min:y_max, x_min:x_max]

            prediction = 1
            if face_crop.size > 0:
                try:
                    resized = cv2.resize(face_crop, (224, 224))
                    normed = resized.astype("float32") / 255.0
                    input_face = np.expand_dims(normed, axis=0)
                    prediction = model.predict(input_face, verbose=0)[0][0]
                except:
                    pass

            # Drowsiness Detection
            if ear < EAR_THRESHOLD and prediction < 0.5:
                st.session_state.eye_closed_counter += 1
                if st.session_state.eye_closed_counter >= CONSEC_FRAMES:
                    label = "😴 Drowsy"
                    color = (0, 0, 255)
                    if st.session_state.beep_count['drowsy'] < 2:
                        play_alert(count=3, frequency=1200)
                        st.session_state.beep_count['drowsy'] += 1
                    log_event(label)
            else:
                st.session_state.eye_closed_counter = 0
                st.session_state.beep_count['drowsy'] = 0

            # Yawning Detection
            top_lip = (lm[13].x * w, lm[13].y * h)
            bottom_lip = (lm[14].x * w, lm[14].y * h)
            left_corner = (lm[78].x * w, lm[78].y * h)
            right_corner = (lm[308].x * w, lm[308].y * h)
            yawn_ratio = euclidean(top_lip, bottom_lip) / euclidean(left_corner, right_corner)

            if yawn_ratio > YAWN_THRESHOLD:
                label = "🗣️ Yawning"
                color = (255, 0, 0)
                if st.session_state.beep_count['yawn'] < 2:
                    play_alert(count=1, frequency=1000)
                    st.session_state.beep_count['yawn'] += 1
                log_event(label)
            else:
                st.session_state.beep_count['yawn'] = 0

            # Head Nodding Detection
            nose_y = lm[1].y * h
            if st.session_state.prev_nose_y is not None:
                diff = st.session_state.prev_nose_y - nose_y
                st.session_state.nod_history.append(diff)
                if len(st.session_state.nod_history) == NOD_HISTORY and sum(st.session_state.nod_history) > NOD_THRESHOLD:
                    label = "🤯 Head Nodding"
                    color = (0, 255, 255)
                    st.session_state.nod_history.clear()
                    if st.session_state.beep_count['nod'] < 2:
                        play_alert(count=2, frequency=900)
                        st.session_state.beep_count['nod'] += 1
                    log_event(label)
                else:
                    st.session_state.beep_count['nod'] = 0
            st.session_state.prev_nose_y = nose_y

        # Display on frame
        cv2.putText(frame, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"EAR: {ear:.2f} | Model: {prediction:.2f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# ==================== LOG VIEWER ====================
st.markdown("---")
with st.expander("📑 Today's Logged Events"):
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        with open("drowsiness_log.txt", "r", encoding="utf-8") as f:
            logs = [line for line in f if line.startswith(today)]
            if logs:
                st.code("".join(logs), language="text")
            else:
                st.success("✅ No events logged today.")
    except FileNotFoundError:
        st.info("No log file found.")
