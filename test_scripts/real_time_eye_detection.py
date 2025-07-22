import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('eye_state_classifier.h5')

#Using image size that was used in training
IMG_SIZE = 224

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Flipping frame to avoid mirrior image
    frame = cv2.flip(frame, 1)

    #Define region og interest (ROI)
    roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)

    #prediction
    pred = model.predict(roi)[0][0]
    label = "Open Eyes" if pred > 0.5 else "Closed Eyes"

    #displaying prediction
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0) if label == "Open Eyes" else (0, 0, 255), 3)

    cv2.imshow("Real-Time Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
