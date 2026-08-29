<div align="center">

# 🎙️ Speech Emotion Recognition

### AI-Powered Speech Emotion Classification using Machine Learning

**An end-to-end Speech Emotion Recognition system built with Python, Scikit-learn, Librosa and Streamlit using the RAVDESS dataset.**

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio-5C3EE8?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br>

![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-8B5CF6?style=flat-square)
![Samples](https://img.shields.io/badge/Samples-1%2C440-06B6D4?style=flat-square)
![Emotions](https://img.shields.io/badge/Emotions-8-EC4899?style=flat-square)
![Features](https://img.shields.io/badge/Audio%20Features-307-10B981?style=flat-square)
![Evaluation](https://img.shields.io/badge/Evaluation-Actor%20Independent-F59E0B?style=flat-square)

<br><br>

<a href="https://github.com/AliZain3311/CodeAlpha_Emotion_Recognition">
  <img src="https://img.shields.io/badge/⭐%20View%20Repository-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
</a>

</div>

---

## 🧠 About The Project

**Speech Emotion Recognition (SER)** is a machine learning task focused on identifying emotional states from human speech.

This project implements a complete end-to-end SER pipeline that transforms raw speech recordings into meaningful acoustic features, trains and evaluates multiple machine learning models, performs feature selection and hyperparameter optimization, and exposes the trained model through a modern **Streamlit web interface**.

The system recognizes **8 different emotional states** from `.wav` speech recordings.

### What makes this project different?

Instead of relying only on a random train/test split, the project uses an **actor-independent evaluation strategy**.

Training and testing speakers are completely separated:

```text
TRAINING ACTORS
Actor_01 → Actor_19
        ↓
     1,140 samples

TESTING ACTORS
Actor_20 → Actor_24
        ↓
       300 samples