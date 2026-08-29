# 🎙️ Speech Emotion Recognition

An end-to-end Speech Emotion Recognition (SER) system built with Python and machine learning using the RAVDESS emotional speech dataset.

The project extracts advanced acoustic features from speech, performs actor-independent evaluation, compares multiple machine learning models, applies feature selection and hyperparameter optimization, and provides a modern Streamlit web interface for real-time audio emotion prediction.

---

## 🚀 Project Highlights

- Speech emotion recognition from `.wav` audio files
- RAVDESS dataset with 1,440 audio samples
- 8 emotion classes
- Actor-independent train/test split
- Advanced audio feature extraction
- MFCC and Delta-MFCC features
- Chroma features
- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Feature selection using SelectKBest
- SVM hyperparameter optimization
- Actor-aware cross-validation
- Confusion matrix and classification reports
- Saved trained ML model
- Streamlit browser-based interface

---

## 🎭 Emotion Classes

The system recognizes the following emotions:

1. Angry
2. Calm
3. Disgust
4. Fearful
5. Happy
6. Neutral
7. Sad
8. Surprised

---

## 📊 Dataset

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) speech dataset.

Dataset characteristics used in this project:

- 1,440 audio samples
- 24 actors
- 8 emotion classes
- WAV audio format

To reduce actor leakage, the dataset was divided by actor rather than randomly splitting individual audio files.

### Training Set

Actors:

`Actor_01` – `Actor_19`

Samples:

`1,140`

### Test Set

Actors:

`Actor_20` – `Actor_24`

Samples:

`300`

No actors are shared between the training and testing sets.

---

## 🧠 Feature Extraction

The audio processing pipeline extracts multiple acoustic characteristics.

### MFCC

40 MFCC coefficients are extracted along with their mean and standard deviation.

### Delta MFCC

First-order temporal changes in MFCC features are calculated.

### Chroma

12 chroma features are extracted to capture pitch-class information.

### Spectral Features

The system also extracts:

- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff

The final feature representation contains 307 numerical audio features.

---

## 🤖 Machine Learning Models

Three baseline models were evaluated:

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Random Forest | 49.0% | 45.1% |
| SVM | 49.0% | 48.9% |
| MLP Neural Network | 46.7% | 46.6% |

SVM provided the strongest baseline performance.

---

## 🔎 Feature Selection

Feature selection was performed using `SelectKBest` with ANOVA F-test scoring.

Different feature subset sizes were evaluated.

The best subset contained:

**100 features**

Performance:

- Accuracy: **52.0%**
- Weighted Precision: **54.0%**
- Weighted Recall: **52.0%**
- Weighted F1: **51.4%**

---

## ⚙️ Advanced Model

The final optimized pipeline combines:

1. StandardScaler
2. SelectKBest
3. RBF SVM

Best parameters:

```text
Number of selected features: 100
SVM C: 3
Kernel: RBF
Gamma: scale
Class weight: None