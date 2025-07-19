# Violence Detection using ResNet50 + BiLSTM

This AI project detects violence in video streams using a hybrid deep learning model combining **ResNet50** and **BiLSTM**, with a simple Tkinter GUI for live interaction and visualization.

---

## 📁 Project Structure

```
Violence-Detection-ResNet50-BiLSTM/
├── app/
│   └── violence_detection_app.py       ← GUI Application
├── train/
│   └── ResNet50_BiLSTM_Training.ipynb  ← Model Training Code
├── models/
│   └── ResNet50_BiLSTM_.pth            ← Pre-trained Model File (remove before GitHub push)
├── assets/
│   └── preview.png                     ← Screenshot of the GUI
├── requirements.txt                    ← Project dependencies
├── .gitignore
└── README.md
```

---

## ✅ Features

- Feature extraction with ResNet50
- Temporal sequence modeling with BiLSTM
- Real-time violence probability prediction
- Automatically saves and displays detected frames
- GUI built with Tkinter
- Supports both video file and live camera input

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
python app/violence_detection_app.py
```

---

## 📦 Requirements

```
torch
torchvision
opencv-python
pillow
tk
numpy
```

---

## 📸 GUI Preview




---

## 👨‍💻 Developer

**Name**: ABDULLAH ALSAYED ALI  
**Role**: Computer Engineering Student  
**Project Type**: Academic Final Project
