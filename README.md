# Sign Language Detection Project

This project is a real-time sign language detection system that uses **MediaPipe** for hand tracking and **TensorFlow** for gesture classification. It can recognize gestures corresponding to the letters A-Z in American Sign Language (ASL). The project is divided into four main modules: data collection (`prepare.py`), data preprocessing (`process.py`), model training (`train.py`), and real-time testing (`test.py`).

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Local Setup (VS Code)](#local-setup-vs-code)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Real-Time Testing](#real-time-testing)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

---

## Overview

The project consists of the following modules:

1. **`prepare.py`**: Collects hand landmark data for gestures (A-Z) using a webcam.
2. **`process.py`**: Preprocesses the collected data and splits it into training and testing sets.
3. **`train.py`**: Trains a neural network model using TensorFlow to classify gestures.
4. **`test.py`**: Performs real-time gesture recognition using the trained model.

---

## Setup

### Local Setup (VS Code)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Amrokahla/custom-sign-Language-detection.git
   cd custom-sign-Language-detection
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate```
   or
   ``` ./venv/Scripts/activate
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the scripts**:
   Follow the instructions in the [Usage](#usage) section.
   
## Usage

### Data Collection

1. Run `prepare.py` to collect gesture data:
   ```bash
   python src/prepare.py
- Press the corresponding key (A-Z) to record a gesture.
- Press Enter to exit and save the data to gesture_data/gesture_data.csv.
### Data Preprocessing

2. Run `process.py` to preprocess the data:
   ```bash
   python src/process.py
- This script splits the data into training and testing sets and saves them as .pkl files.

### Model Training
3. Run `train.py` to train the model:
   ```bash
   python src/train.py
- The trained model is saved as gesture_data/model.h5.

### Real-time testing
4. Run `test.py` for real-time gesture recognition:
   `test.py` to test the model:
   ```bash
   python src/test.py
- The webcam feed will display the detected gesture in real-time.

## Dependencies
- Python 3.8+
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- TensorFlow (tensorflow)
- Scikit-learn (scikit-learn)
- Pandas (pandas)
- NumPy (numpy)
Install all dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeatureName).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeatureName).
5. Open a pull request.
