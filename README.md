# AI-Powered Medical Image Analysis System

## Overview
This project is an AI-based medical image classification system that detects diseases from chest X-ray images using deep learning. It classifies images into:
- Normal
- Pneumonia

It also includes a Streamlit web application for real-time predictions.

---

## Problem Statement
Manual diagnosis of X-ray images is time-consuming and prone to human error. This system helps automate early detection using AI.

---

## Solution
A deep learning model (MobileNetV2) trained on chest X-ray images that:
- Extracts features automatically
- Classifies medical images
- Provides real-time predictions via web app

---

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## Dataset
Chest X-ray dataset (Kaggle):
- Normal images
- Pneumonia images

---

## Model Architecture
- Transfer Learning: MobileNetV2
- Global Average Pooling
- Dense Layers
- Softmax Output

---

## Features
- Medical image classification
- Real-time prediction system
- Confidence score output
- Web-based UI using Streamlit
- Confusion matrix evaluation

---

## Results
- Accuracy: ~85–95% (depends on training)
- Confusion Matrix generated
- Real-time predictions

---

## How to Run

### 1. Install dependencies
### 2. Run Streamlit app

---

##  Project Structure
AI-Medical-Image-Analysis-System/
│
├── data/
├── models/
├── src/
├── app.py
├── requirements.txt
├── README.md


---

## Output Example
- Input: Chest X-ray image
- Output: Pneumonia / Normal
- Confidence: 86% - 95%

---

## Author
Bindu P

---

## Future Improvements
- Add more diseases
- Improve dataset size
- Deploy on cloud (AWS / Render)
- Add Grad-CAM explainability
