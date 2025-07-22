import cv2
import winsound

# Load Haar cascade for eyes
EYE_CASCADE_PATH = "../models/haarcascade_eye.xml"
THRESHOLD_FRAMES = 20

def load_eye_detector(path):
    return cv2.CascadeClassifier(path)

def detect_eyes(eye_cascade, gray_frame):
    return eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

def main():
    eye_cascade = load_eye_detector(EYE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    eye_closed_frames = 0

    if not cap.isOpened():
        print("Could not open Webcam")
        return 
    
    print("Webcam opened starting detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = detect_eyes(eye_cascade, gray)

        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0  # reset if eyes are found

    # Trigger warning if eyes closed for too long
        if eye_closed_frames >= THRESHOLD_FRAMES:
            cv2.putText(frame, "Drowsiness Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            winsound.Beep(1000, 300)

    # Draw rectangles around eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
