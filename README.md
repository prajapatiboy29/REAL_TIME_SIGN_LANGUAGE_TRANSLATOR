# Real-Time Sign Language Translator

A Python-based real-time sign language recognition system that uses deep learning, hand landmark detection, and a desktop UI to recognize hand signs from webcam input and display predictions live.

## Overview

This project was developed to support sign language recognition in real time using computer vision and machine learning. It combines two approaches:

- **CNN-based image classification** for dataset-specific ASL and ISL model training
- **Landmark-based real-time recognition** using MediaPipe and a Random Forest classifier for more stable live alphabet prediction

A Tkinter-based desktop interface is included for easier use and final project demonstration.

## Objectives

- Build a real-time sign language recognition system using webcam input
- Train separate ASL and ISL image-based CNN models
- Improve real-time prediction using hand landmarks
- Provide a simple desktop user interface for live interaction
- Create a modular system that can be extended further

## Features

- Real-time webcam-based recognition
- ASL CNN training pipeline
- ISL CNN training pipeline
- Landmark-based alphabet recognition
- ASL / ISL mode selection in UI
- Live prediction and confidence display
- Tkinter-based desktop interface
- Structured project layout for future enhancements

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Tkinter
- Visual Studio Code

## Project Structure

```text
Real-time sign Language Tanslator/
├── data/
│   ├── asl_alphabet_train/
│   ├── asl_alphabet_test/
│   ├── ISL/
│   └── landmarks/
│       ├── alphabet_landmarks.csv
│       └── isl_landmarks.csv
├── models/
│   ├── asl_model.h5
│   ├── asl_class_names.json
│   ├── isl_model.h5
│   ├── isl_class_names.json
│   ├── alphabet_landmark_model.pkl
│   └── alphabet_landmark_labels.json
├── report_assets/
├── src/
│   ├── app_ui.py
│   ├── collect_asl_landmarks.py
│   ├── collect_isl_landmarks.py
│   ├── realtime_asl_isl_landmark.py
│   ├── realtime_asl_landmark.py
│   ├── realtime_cnn_translator.py
│   ├── train_asl_cnn_model.py
│   ├── train_asl_landmark_model.py
│   ├── train_isl_cnn_model.py
│   └── train_isl_landmark_model.py
├── requirements.txt
└── README.md
Datasets
ASL Dataset
ASL Alphabet - Kaggle
ISL Dataset
Indian Sign Language Alphabet Dataset (A–Z) - Kaggle
Modules
1. ASL CNN Model
This model is trained using the ASL alphabet image dataset and performs image-based classification of sign gestures.

2. ISL CNN Model
This model is trained separately on the ISL image dataset to support Indian Sign Language recognition.

3. Landmark-Based Alphabet Recognizer
This module uses MediaPipe hand landmarks and a Random Forest classifier for more stable live alphabet prediction.

4. Shared ASL / ISL Landmark Demo
For the final demo, a shared static alphabet landmark model is used in both ASL and ISL modes. Separate CNN models are still maintained for ASL and ISL dataset-specific training.

5. Tkinter UI
The UI allows the user to:

choose ASL or ISL mode
start recognition
stop recognition
view live prediction and confidence
exit the application
Installation
1. Create a virtual environment
python -m venv .venv
2. Activate the virtual environment
Windows PowerShell:

.\.venv\Scripts\Activate
3. Install dependencies
pip install -r requirements.txt
Recommended Requirements
tensorflow==2.16.1
opencv-python==4.11.0.86
mediapipe==0.10.14
numpy==1.26.4
pandas
matplotlib==3.8.4
pillow==11.1.0
protobuf==4.25.3
scikit-learn
Training Commands
Train ASL CNN model
python src/train_asl_cnn_model.py
Train ISL CNN model
python src/train_isl_cnn_model.py
Collect ASL landmark data
python src/collect_asl_landmarks.py
Train ASL landmark model
python src/train_asl_landmark_model.py
Collect ISL landmark data
python src/collect_isl_landmarks.py
Train ISL landmark model
python src/train_isl_landmark_model.py
Running the Project
Main UI
python src/app_ui.py
Optional landmark recognizer
python src/realtime_asl_isl_landmark.py
Optional ASL landmark-only recognizer
python src/realtime_asl_landmark.py
Optional CNN-based recognizer
python src/realtime_cnn_translator.py
Model Performance
ASL CNN Model
Training Accuracy: 99.05%
Validation Accuracy: 99.80%
ISL CNN Model
Training Accuracy: 85.78%
Validation Accuracy: 88.85%
Landmark Alphabet Model
Accuracy: approximately 93%
Working Principle
The webcam captures live video frames.
MediaPipe detects the hand and extracts landmarks.
Landmark coordinates are converted into structured features.
The trained model predicts the sign label.
The UI displays the predicted sign and confidence score.
Parameters
CNN Parameters
Input image size: 64 × 64 × 3
Batch size: 32
Epochs: 10
Train/validation split: 80:20
Dropout rate: 0.3
Landmark Parameters
Landmarks per hand: 21
Coordinates per landmark: x, y, z
Total features: 126
Hyperparameters
CNN Hyperparameters
Optimizer: Adam
Loss function: Sparse Categorical Cross-Entropy
Activation: ReLU
Output activation: Softmax
Filters: 32, 64, 128
Kernel size: 3 × 3
Landmark Model Hyperparameters
Classifier: Random Forest
Number of estimators: 200–300
Random state: 42
Limitations
The system works better for static signs than dynamic gestures
Real-time accuracy depends on lighting, background, and hand angle
Some alphabets may still be confused during live recognition
The shared landmark recognizer is currently a prototype demo module
Full sequence-based dynamic sign recognition is not implemented yet
Future Scope
Full separate landmark-based ISL recognizer
Dynamic gesture recognition using LSTM or GRU
Two-hand sign recognition
Word and sentence formation
Text-to-speech output
Better UI with embedded webcam feed
Deployment as a standalone desktop or web application
References
K. Sinha et al., “Gesture Translation and Communication Improvement using ISL dataset (A–Z, 1–9),” 2022.
M. Sanaullah et al., “Text-to-Sign Conversion System for Accessibility Enhancement,” 2022.
M. Papatsimouli et al., “Survey of Real-Time Systems and IoT-Based Integration for Sign Language Translation,” 2023.
S. Das et al., “Gesture Recognition using Hybrid CNN-BiLSTM Model with Key-Frame Extraction,” 2023.
B. Chempavathy et al., “Language Translation and Performance Optimization using Hybrid Attention and Quaternion Neural Networks,” 2024.
S. R. Patil et al., “Vision-Based Gesture Interpretation System with Skeletal Tracking and ML Recognition,” 2025.
M. Geetha et al., “Continuous Sign Language Recognition using SignFlow Model with RGB and Pose Data,” 2025.
B. Alsharif et al., “Real-Time American Sign Language Interpretation using Deep Learning and Keypoint Tracking,” 2025.
Gupta, M. (2022). Survey on sign language recognition in context of vision-based and deep learning. Measurement: Sensors, 100385. 
https://doi.org/10.1016/j.measen.2022.100385
Argade, D., et al. (2025). Sign Language Gesture Detection Using CNN. International Journal for Research in Applied Science and Engineering Technology (IJRASET). 
https://doi.org/10.22214/ijraset.2025.69577
Madhukar, B. N., et al. (2025). Real-Time Sign Language Recognition and Translation: A Survey of Deep Learning Techniques. International Journal for Research in Applied Science and Engineering Technology (IJRASET).
F. Zhang et al., “MediaPipe Hands: On-Device Real-Time Hand Tracking,” CV4ARVR 2020. Available: 
https://research.google/pubs/mediapipe-hands-on-device-real-time-hand-tracking/
Akash, “ASL Alphabet,” Kaggle dataset. Available: 
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Priyavshah, “Indian Sign Language Alphabet Dataset (A–Z),” Kaggle dataset. Available: 
https://www.kaggle.com/datasets/priyavshah/indian-sign-language-alphabet-dataset-az
Author
Hemant

License
This project is intended for learning, academic, and educational use.


If you want, I can also make this into a **more professional GitHub version** with badges and a cleaner public-project style.
