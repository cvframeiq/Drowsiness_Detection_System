# Drowsiness_Detection_System

🚀 Built by Mitali under FrameIQ LLP | Updated: August 2025  
⚙️ Tech Stack: Flask, OpenCV, MediaPipe, TensorFlow, HTML/CSS

---

## 🔍 Overview

**Drowsiness Detection System** is an AI-powered, real-time monitoring solution that detects fatigue signs—like eye closure, yawning, and head nodding—using a hybrid combination of **Computer Vision** and **Deep Learning**.  
This lightweight system runs locally through a **Flask web interface**, providing:

- 🎥 Live webcam feed  
- 🧠 Eye Aspect Ratio (EAR) + CNN predictions  
- 🔔 Sound alerts  
- 📜 Automatic event logging  
- 🟢 Dynamic status updates in-browser

> Perfect for driver monitoring, workplace safety, productivity research, or academic/portfolio showcases.

---

## 🧠 Key Features

- 🧍 MediaPipe Facial Landmark Detection  
- 👁️ Eye State Classification via CNN (MobileNetV2)  
- 💤 Drowsiness Detection using EAR + ML logic  
- 😮 Yawning & Head Nodding Detection  
- 🛜 Real-time Flask UI (No Streamlit required)  
- 🔊 Sound Alerts + Timestamped Logging  
- 📊 Live Status: Attentive / Drowsy / Yawning / Head Nodding  
- ✅ Clean, modular folder structure  

---

## 🗂️ Project Structure
```
Drowsiness_Detection_System/
├── app.py                          # Flask backend (UI + detection logic)
├── templates/
│   └── index.html                  # Responsive HTML frontend
├── model_training/
│   ├── preprocessing_modeltraining.py
│   ├── eye_state_classifier.h5
│   └── eye_state_classifier.keras
├── test_scripts/
│   └── test_model_prediction.py
├── streamlit_version_archived/     # Previous Streamlit version (archived)
│   ├── app.py
│   └── drowsiness_log.txt
├── drowsiness_log.txt              # Auto-generated event log file
├── requirements.txt                # All dependencies
├── README.md                       # You’re reading it!
└── .gitignore                      # Version control hygiene
```

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/cvframeiq/Drowsiness_Detection_System.git
cd Drowsiness_Detection_System

# 2. Create a virtual environment
python -m venv cv_env
cv_env\Scripts\activate     # On Windows
# source cv_env/bin/activate   # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
python app.py
```
Open your browser at http://localhost:5000 to see the app in action.


---

## 📊 Model Details

**Architecture:** MobileNetV2 (Transfer Learning)  
**Input Size:** 224 x 224  
**Classes:** Binary (Open / Closed Eyes)  
**Dataset:** CEW – Closed Eyes in the Wild  
**Accuracy:** ~87% (Training) | ~76% (Validation)

---

## 🛡️ Security Notes

✅ Webcam access only on local system  
✅ No image/video is saved  
✅ No external API calls — full offline execution

---

## 📦 Dataset

> **Note:** Dataset not included due to size. Please create the following folder structure:

```
dataset/
├── open/
│   └── *.jpg
└── closed/
    └── *.jpg
```

📌 Archived: Streamlit Version
The earlier version using Streamlit has been moved to /streamlit_version_archived for reference.

