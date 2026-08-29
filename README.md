# 🎙️ Speech Emotion Recognition

> **An end-to-end Speech Emotion Recognition (SER) system built with Python, Machine Learning, and Streamlit using the RAVDESS emotional speech dataset.**

This project develops a complete speech emotion recognition pipeline — from raw audio preprocessing and advanced acoustic feature extraction to actor-independent evaluation, model comparison, feature selection, hyperparameter optimization, and an interactive browser-based prediction interface.

The system can analyze `.wav` speech recordings and predict one of **8 different emotional states** using a trained machine learning pipeline.

---

## ✨ Project Highlights

- 🎙️ Speech emotion recognition from `.wav` audio
- 🎭 **8 emotion classes**
- 🎧 **1,440 RAVDESS audio samples**
- 👥 Actor-independent train/test evaluation
- 🔊 Advanced acoustic feature extraction
- 🧠 MFCC & Delta-MFCC features
- 🎵 Chroma features
- 📈 Spectral audio features
- 🔎 SelectKBest feature selection
- ⚙️ SVM hyperparameter optimization
- 🧪 Actor-aware cross-validation
- 📊 Confusion matrix & classification reports
- 💾 Saved trained ML models
- 🌐 Modern Streamlit web application
- 🚀 Real-time audio emotion prediction

---

## 🎭 Supported Emotions

The system recognizes **8 emotional categories**:

| Emotion | Description |
|---|---|
| 😠 Angry | Anger / frustration |
| 😌 Calm | Calm / relaxed speech |
| 🤢 Disgust | Disgusted emotional state |
| 😨 Fearful | Fear / anxiety |
| 😊 Happy | Happiness / positive emotion |
| 😐 Neutral | Neutral emotional state |
| 😢 Sad | Sadness / low-energy emotion |
| 😲 Surprised | Surprise / unexpected reaction |

---

## 📊 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** speech dataset.

### Dataset Configuration

- **1,440 audio samples**
- **24 actors**
- **8 emotion classes**
- **WAV audio format**

To prevent **actor leakage**, the dataset was divided by actor rather than randomly splitting individual audio files.

### 🏋️ Training Set

**Actors:** `Actor_01` – `Actor_19`

**Samples:** `1,140`

### 🧪 Test Set

**Actors:** `Actor_20` – `Actor_24`

**Samples:** `300`

> ✅ **No actors are shared between the training and testing sets.**

This provides a more realistic evaluation of how the model performs on **unseen speakers**.

---

## 🎧 Audio Feature Extraction

The project extracts a rich set of acoustic features from every audio recording.

### 🧠 MFCC

**40 MFCC coefficients** are extracted along with their statistical representations.

MFCCs capture important characteristics of the speech spectrum and are widely used in speech-related machine learning tasks.

### 📈 Delta MFCC

First-order temporal changes in MFCC features are calculated to capture how speech characteristics evolve over time.

### 🎵 Chroma Features

**12 chroma features** are extracted to represent pitch-class information.

### 🔊 Spectral Features

Additional acoustic characteristics include:

- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff

### 📦 Final Feature Representation

The final processed dataset contains:

**307 numerical audio features per sample.**

---

## 🤖 Machine Learning Models

Three baseline machine learning models were evaluated:

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| 🌲 Random Forest | 49.0% | 45.1% |
| ⚡ SVM | 49.0% | 48.9% |
| 🧠 MLP Neural Network | 46.7% | 46.6% |

### 🏆 Best Baseline

**Support Vector Machine (SVM)** provided the strongest baseline performance based on weighted F1 score.

---

## 🔎 Feature Selection

Feature selection was performed using:

**SelectKBest + ANOVA F-test**

Multiple feature subset sizes were evaluated:

- 50 features
- 100 features
- 150 features
- 200 features
- 250 features
- 300 features

### 🏆 Best Feature Subset

The best-performing configuration selected:

**100 features**

Performance on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | **52.0%** |
| Weighted Precision | **54.0%** |
| Weighted Recall | **52.0%** |
| Weighted F1 | **51.4%** |

Feature selection improved the weighted F1 score compared with the original SVM baseline.

---

## ⚙️ Advanced Model Optimization

The optimized emotion recognition pipeline combines:

```text
Audio
   ↓
Feature Extraction
   ↓
StandardScaler
   ↓
SelectKBest
   ↓
RBF SVM
   ↓
Emotion Prediction