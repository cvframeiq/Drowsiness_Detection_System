import cv2
import os

print("Eye Detection Script Started...")

model_path = "../models/haarcascade_eye.xml"
print("Looking for model at:", os.path.abspath(model_path))
print("Model exists?", os.path.exists(model_path))

eye_cascade = cv2.CascadeClassifier(model_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("Webcam opened!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit pressed.")
        break

cap.release()
cv2.destroyAllWindows()
print("Stream closed cleanly.")
