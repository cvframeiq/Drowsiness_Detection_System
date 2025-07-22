import cv2
import mediapipe as mp
import math
import winsound

# Constants
THRESHOLD_EAR = 0.21
CLOSED_FRAMES_THRESHOLD = 20

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
draw_utils = mp.solutions.drawing_utils

# Eye landmarks (left and right)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_EAR(landmarks, eye_indices, img_w, img_h):
    def get_point(i): return (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))

    p1, p2, p3, p4, p5, p6 = [get_point(i) for i in eye_indices]
    dist_vert1 = math.dist(p2, p6)
    dist_vert2 = math.dist(p3, p5)
    dist_horiz = math.dist(p1, p4)
    ear = (dist_vert1 + dist_vert2) / (2.0 * dist_horiz)
    return ear

cap = cv2.VideoCapture(0)
closed_frames = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark

        left_ear = calculate_EAR(mesh_points, LEFT_EYE, img_w, img_h)
        right_ear = calculate_EAR(mesh_points, RIGHT_EYE, img_w, img_h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < THRESHOLD_EAR:
            closed_frames += 1
        else:
            closed_frames = 0

        # Draw status
        if closed_frames >= CLOSED_FRAMES_THRESHOLD:
            cv2.putText(frame, "😴 Drowsiness Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            winsound.Beep(1000, 200)

    cv2.imshow("Drowsiness Detection (EAR)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
