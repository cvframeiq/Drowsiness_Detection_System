from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
import threading
import time
import datetime
import winsound
import tensorflow as tf
import mediapipe as mp
import math
from collections import deque

app = Flask(__name__)

# ==================== MODEL + MEDIAPIPE ====================
model = tf.keras.models.load_model('model_training/eye_state_classifier.keras')
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

# ==================== GLOBALS ====================
cap = None
frame_lock = threading.Lock()
detection_active = False
video_active = False
frame_to_send = None

status = "Idle"
eye_closed_counter = 0
nod_history = deque(maxlen=10)
prev_nose_y = None
beep_count = {'drowsy': 0, 'yawn': 0, 'nod': 0}

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

def get_today_logs():
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        with open("drowsiness_log.txt", "r", encoding="utf-8") as f:
            return [line for line in f if line.startswith(today)]
    except FileNotFoundError:
        return []

# ==================== STREAMING & DETECTION ====================
def detection_loop():
    global frame_to_send, detection_active, cap, status, eye_closed_counter, prev_nose_y, beep_count

    while video_active:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        label = "🟢 Attentive"
        color = (0, 255, 0)
        ear, prediction = 0.3, 1.0

        if detection_active and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_eye = [(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE_IDX]
            right_eye = [(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE_IDX]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            x_min = int(min([p.x for p in lm]) * w)
            x_max = int(max([p.x for p in lm]) * w)
            y_min = int(min([p.y for p in lm]) * h)
            y_max = int(max([p.y for p in lm]) * h)
            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size > 0:
                try:
                    resized = cv2.resize(face_crop, (224, 224))
                    normed = resized.astype("float32") / 255.0
                    input_face = np.expand_dims(normed, axis=0)
                    prediction = model.predict(input_face, verbose=0)[0][0]
                except:
                    pass

            # Drowsy
            if ear < EAR_THRESHOLD and prediction < 0.5:
                eye_closed_counter += 1
                if eye_closed_counter >= CONSEC_FRAMES:
                    label = "😴 Drowsy"
                    color = (0, 0, 255)
                    if beep_count['drowsy'] < 1:
                        play_alert(count=4, frequency=1200)
                        beep_count['drowsy'] += 1
                    log_event(label)
            else:
                eye_closed_counter = 0
                beep_count['drowsy'] = 0

            # Yawning
            top_lip = (lm[13].x * w, lm[13].y * h)
            bottom_lip = (lm[14].x * w, lm[14].y * h)
            left_corner = (lm[78].x * w, lm[78].y * h)
            right_corner = (lm[308].x * w, lm[308].y * h)
            yawn_ratio = euclidean(top_lip, bottom_lip) / euclidean(left_corner, right_corner)

            if yawn_ratio > YAWN_THRESHOLD:
                label = "🗣️ Yawning"
                color = (255, 0, 0)
                if beep_count['yawn'] < 1:
                    play_alert(count=2, frequency=1000)
                    beep_count['yawn'] += 1
                log_event(label)
            else:
                beep_count['yawn'] = 0

            # Head Nodding
            nose_y = lm[1].y * h
            if prev_nose_y is not None:
                diff = prev_nose_y - nose_y
                nod_history.append(diff)
                if len(nod_history) == NOD_HISTORY and sum(nod_history) > NOD_THRESHOLD:
                    label = "🤯 Head Nodding"
                    color = (0, 255, 255)
                    nod_history.clear()
                    if beep_count['nod'] < 1:
                        play_alert(count=1, frequency=900)
                        beep_count['nod'] += 1
                    log_event(label)
                else:
                    beep_count['nod'] = 0
            prev_nose_y = nose_y

        status = label
        cv2.putText(frame, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        with frame_lock:
            frame_to_send = cv2.imencode('.jpg', frame)[1].tobytes()

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', logs=get_today_logs(), status=status)

@app.route('/video_feed')
def video_feed():
    def generate():
        global frame_to_send
        while True:
            with frame_lock:
                if frame_to_send:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global cap, video_active
    if not video_active:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  
        video_active = True
        threading.Thread(target=detection_loop, daemon=True).start()
    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
def stop():
    global video_active, cap
    video_active = False
    if cap:
        cap.release()
    return redirect(url_for('index'))

@app.route('/detect', methods=['POST'])
def detect():
    global detection_active
    detection_active = True
    return redirect(url_for('index'))

@app.route('/get_status')
def get_status():
    return jsonify({'status': status})


# ==================== MAIN ====================
if __name__ == '__main__':
    app.run(debug=True)
