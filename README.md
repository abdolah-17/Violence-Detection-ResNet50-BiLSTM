# Violence Detection using ResNet50 + BiLSTM

This project aims to detect violent scenes in video clips using a deep learning model that combines ResNet50 for feature extraction and BiLSTM for temporal sequence analysis.

## Project Contents

- `ResNet50_BiLSTM_Training.ipynb`: Jupyter notebook for training the model
- `violence_detection_app.py`: Script to run the trained model on new videos
- `requirements.txt`: List of required Python libraries
- `.gitignore`: Excludes unnecessary files (e.g., model weights, cache)

## How to Use

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (optional if model is already available):

   - Open `ResNet50_BiLSTM_Training.ipynb` in Jupyter Notebook
   - Run all cells to train and save the model

3. Run the detection app:

   ```bash
   python violence_detection_app.py
   ```

   Make sure to set the correct path to the video inside the script before running.

## Notes

- The `.pth` model file is **not included** in this repository
- You can retrain the model using the provided notebook
- Supported video formats include `.mp4`, `.avi`, etc.

## Folder Structure

```
Violence-Detection-ResNet50-BiLSTM/
│
├── README.md
├── requirements.txt
├── ResNet50_BiLSTM_Training.ipynb
├── violence_detection_app.py
├── .gitignore
└── Violence-Detection-ResNet50-BiLSTM/   ← nested folder (currently unused)
```

## Author

Developed by Abdullah as part of an academic project on detecting violence in video content.
