# Smart Emergency Vehicle Detection via CNN-Based Acoustic Analysis

An end-to-end sound alert system designed to detect and classify emergency vehicle sirens (ambulance, police, fire) from audio input using Deep Learning and Signal Processing.

## 🚀 Overview
This project uses **1D Convolutional Neural Networks (CNN)** and **MFCC feature extraction** to identify emergency sirens from raw audio data. Built with **Django**, it provides a user-friendly interface for real-time sound classification, model training, and performance visualization.

## ✨ Features
- **Real-time Prediction:** Upload audio files and get instant siren classification.
- **Deep Learning Model:** 1D-CNN architecture optimized for acoustic feature extraction.
- **Signal Processing:** Utilizes MFCC (Mel-frequency cepstral coefficients) via Librosa for high-fidelity audio preprocessing.
- **Admin Dashboard:** Manage users and monitor training results (Accuracy/Loss curves, Confusion Matrices).
- **Secure Authentication:** Built-in registration and login system.

## 🛠️ Tech Stack
- **Backend:** Python, Django
- **Deep Learning:** TensorFlow (Keras), Scikit-learn
- **Audio Processing:** Librosa
- **Data Science:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Database:** SQLite3

## 📂 Project Structure
```text
├── admins/         # Admin functionalities
├── assets/         # Static files (CSS, JS) and HTML Templates
├── media/          # Uploaded audio and generated plots (Ignored by Git)
├── users/          # User-side views and model logic
├── manage.py       # Django management script
└── requirements.txt # Project dependencies
```

## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/siren-sound-alert-system.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
4. **Start the server:**
   ```bash
   python manage.py runserver
   ```
   Visit `http://127.0.0.1:8000` in your browser.

## 📊 Performance
The system generates automated evaluation reports after every training session, including:
- **Accuracy & Loss Curves**
- **Confusion Matrix** for multi-class identification

---
Developed as a Major Project for college.
