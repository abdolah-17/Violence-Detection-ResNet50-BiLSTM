# 🎯 Violence Detection using ResNet50 + BiLSTM

This project detects violent scenes in video clips using a hybrid deep learning model that combines **ResNet50** for spatial feature extraction and **BiLSTM** for temporal sequence modeling.

---

## 📁 Project Structure

```
Violence-Detection-ResNet50-BiLSTM/
├── violence_detection_app.py           ← GUI application using Tkinter
├── ResNet50_BiLSTM_Training.ipynb      ← Model training notebook
├── requirements.txt                    ← Required libraries
├── README.md
├── .gitignore
└── models/                             ← Place to store trained model file
```

---

## 🚀 How to Run

1. **Install required libraries**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (optional)**  
   Open the notebook in Jupyter and run all cells to train the model:
   ```
   ResNet50_BiLSTM_Training.ipynb
   ```

3. **Run the detection application**

   ```bash
   python violence_detection_app.py
   ```

   You can analyze a video or use the live camera from the GUI.

---

## 📦 Download Pretrained Model

You can download the pretrained model file (`ResNet50_BiLSTM_.pth`) from Google Drive:

👉 [Download Model from Google Drive](https://drive.google.com/uc?id=1Nj0vQhunydoMF-zIgAbggOsAoxZ_3HY0&export=download)

After downloading, place the `.pth` file in the `models/` directory and update the path in the script if needed.

---

## 📝 Notes

- Only 16 frames are used per detection window
- The model predicts the probability of violence in each segment
- Detected frames are saved and visualized automatically
- Requires a GPU for best performance, but also works on CPU

---

## 👨‍💻 Developer

**Name**: ABDULLAH ALSAYED ALI  
**Role**: Computer Engineering Student  
**Project**: University Final Project for Real-time Violence Detection
