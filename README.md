# Cat vs Dog Image Classification using SVM

This project implements a **Support Vector Machine (SVM)** model to classify images of cats and dogs using the [Kaggle Cats and Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data).  
Additionally, a **webcam-based application** allows real-time predictions using the trained model.

---

## Features

- Train an **SVM classifier** on the Kaggle Cats & Dogs dataset.
- Preprocess images: resize, flatten, and scale features for SVM input.
- Save and load the trained model using `joblib`.
- **Webcam inference script** for live cat/dog classification.
- Displays predictions in real-time with webcam feed.

---

## Requirements

- Python 3.8+
- Libraries:

```bash
  pip install numpy pandas scikit-learn opencv-python joblib matplotlib
```

## 📂 Project Structure

```bash

PRODIGY_ML_02/
├── svm_resnet.pkl                 # Trained SVM model
├── 03_CatVsDogs.ipynb             # Script to train SVM on Kaggle dataset
├── 03_CatVsDogs_Webcam.py         # Webcam-based prediction script
├── README.md                      # Project documentation

```

## How to Run Webcam Prediction

```bash
Make sure your webcam is connected.

Run the inference script:

python 04_Hand_Gesture_Webcam.py


A window will appear showing the webcam feed and the predicted class (Cat or Dog) for the detected image frame.
```
