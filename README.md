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

       🎭 Emotion Classes

The model recognizes the following emotional categories:

Emotion	Meaning
😠 Angry	Anger / frustration
😌 Calm	Calm / relaxed speech
🤢 Disgust	Disgusted emotional state
😨 Fearful	Fear / anxiety
😊 Happy	Happiness / positive emotion
😐 Neutral	Neutral emotional state
😢 Sad	Sadness / low-energy emotion
😲 Surprised	Surprise / unexpected reaction
📊 Dataset

This project uses the RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song speech dataset.

Dataset Statistics
Property	Value
🎧 Audio Samples	1,440
👥 Actors	24
🎭 Emotion Classes	8
📁 Audio Format	WAV
🔢 Final Features	307
🔐 Actor-Independent Evaluation

A major focus of this project is avoiding actor leakage.

Instead of randomly splitting individual recordings, the dataset was divided by actor.

🏋️ Training
Actor_01 – Actor_19
1,140 samples
🧪 Testing
Actor_20 – Actor_24
300 samples
✅ Leakage Check
Training Actors ∩ Testing Actors = ∅

No actor appears in both training and testing sets.

This means the final test set represents completely unseen speakers.

🎧 Audio Feature Engineering

Raw audio cannot be directly supplied to the classical machine learning models used in this project.

Therefore, each audio recording is transformed into a numerical representation.

The feature extraction pipeline includes:

Raw WAV Audio
      ↓
Audio Loading
      ↓
Signal Processing
      ↓
MFCC
      ↓
Delta MFCC
      ↓
Chroma
      ↓
Spectral Features
      ↓
Statistical Aggregation
      ↓
307 Numerical Features
🧠 MFCC Features

40 Mel-Frequency Cepstral Coefficients (MFCCs) are extracted to capture important characteristics of the speech spectrum.

Statistical representations are then calculated to create a stable feature representation for machine learning.

📈 Delta MFCC

Delta features capture temporal changes in the MFCC representation.

This helps the model understand how speech characteristics evolve throughout an audio recording.

🎵 Chroma Features

12 chroma features are extracted to represent pitch-class information within the audio signal.

🔊 Spectral Features

The pipeline also includes:

Zero Crossing Rate
RMS Energy
Spectral Centroid
Spectral Bandwidth
Spectral Rolloff
🤖 Machine Learning Experiments

Three baseline machine learning models were evaluated.

Model	Accuracy	Weighted F1
🌲 Random Forest	49.0%	45.1%
⚡ SVM	49.0%	48.9%
🧠 MLP Neural Network	46.7%	46.6%
🏆 Best Baseline

The Support Vector Machine (SVM) produced the strongest baseline weighted F1 score.

🔎 Feature Selection

Feature selection was performed using:

SelectKBest
     +
ANOVA F-test

The following feature subset sizes were evaluated:

50
100
150
200
250
300
🏆 Best Feature Subset

The best-performing configuration used:

100 selected features

Performance
Metric	Score
Accuracy	52.0%
Weighted Precision	54.0%
Weighted Recall	52.0%
Weighted F1	51.4%

The feature-selected SVM improved over the original SVM baseline in weighted F1 performance.

⚙️ Advanced Model Optimization

The advanced pipeline combines preprocessing, feature selection and an optimized SVM classifier.

                 ┌──────────────────┐
                 │    WAV Audio      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Feature Extraction│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ 307 Audio Features│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │  StandardScaler  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   SelectKBest    │
                 │    100 Features  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │    RBF SVM       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Emotion Prediction│
                 └──────────────────┘
🔧 Optimized Configuration
Selected Features : 100
SVM C              : 3
Kernel             : RBF
Gamma              : scale
Class Weight       : None
🧪 Actor-Aware Cross-Validation

Hyperparameter selection was performed using actor-aware cross-validation.

This helps ensure that model selection does not depend on seeing recordings from the same speakers during validation.

Cross-Validation Result
Actor-Aware CV F1
        ↓
     0.5020
🏁 Final Held-Out Test Performance

The final test set remained completely untouched during feature selection and hyperparameter tuning.

Final Results
Metric	Score
🎯 Accuracy	51.0%
Precision	51.3%
Recall	51.0%
Weighted F1	50.3%
Macro F1	48.3%
Important Evaluation Note

The model was evaluated on completely unseen actors.

Speech emotion recognition is challenging because emotional expression can vary significantly across:

Speakers
Speaking styles
Recording conditions
Emotional intensity
Individual vocal characteristics

Therefore, the actor-independent result provides a more meaningful estimate of real-world generalization than a random file-level split.

📋 Final Classification Performance
Emotion	Precision	Recall	F1
Angry	0.45	0.70	0.55
Calm	0.60	0.38	0.46
Disgust	0.56	0.62	0.59
Fearful	0.61	0.55	0.58
Happy	0.51	0.45	0.48
Neutral	0.25	0.15	0.19
Sad	0.35	0.38	0.36
Surprised	0.64	0.68	0.66
📊 Evaluation Visualizations

The project generates visual evaluation artifacts including:

Confusion Matrix

The confusion matrix provides a class-by-class view of the model's predictions and helps identify which emotions are most frequently confused.

🌐 Interactive Streamlit Application

A browser-based interface was developed using Streamlit.

The application allows users to interact with the trained model without writing Python code.

Application Workflow
Upload WAV
     ↓
Audio Processing
     ↓
Feature Extraction
     ↓
Feature Selection
     ↓
SVM Inference
     ↓
Predicted Emotion
     ↓
Confidence & Probabilities
▶️ Start the Application
streamlit run app.py

Then open the local URL displayed by Streamlit.

🖥️ Application Preview

Add your best Streamlit screenshots here after capturing them.

🎙️ Prediction Interface
Upload a WAV file
        ↓
Run prediction
        ↓
View detected emotion
        ↓
View confidence
        ↓
View emotion probabilities
📁 Project Structure
CodeAlpha_Emotion_Recognition/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── advanced_model_training.py
│   ├── eda.py
│   ├── evaluate_feature_selected.py
│   ├── evaluate_model.py
│   ├── explore_dataset.py
│   ├── extract_features.py
│   ├── feature_selection.py
│   ├── improve_model.py
│   ├── prepare_data.py
│   ├── train_cnn.py
│   ├── train_models.py
│   ├── tune_svm.py
│   └── predict.py
│
├── data/
│   ├── train_features.csv
│   ├── test_features.csv
│   ├── features.csv
│   ├── model_results.csv
│   ├── svm_tuning_results.csv
│   ├── feature_selection_results.csv
│   ├── advanced_model_cv_results.csv
│   ├── advanced_model_results.csv
│   └── advanced_confusion_matrix.png
│
├── models/
│   ├── advanced_emotion_model.joblib
│   ├── final_svm_pipeline.joblib
│   ├── feature_selected_svm.joblib
│   ├── tuned_svm.joblib
│   ├── best_model.joblib
│   └── ...
│
├── emotion_distribution.png
├── mfcc_visualization.png
├── waveform.png
├── confusion_matrix.png
└── cnn_confusion_matrix.png
🛠️ Technology Stack
<div align="center">
Technology	Role
🐍 Python	Core development
🎧 Librosa	Audio processing & feature extraction
🔢 NumPy	Numerical computation
🐼 Pandas	Data processing
🤖 Scikit-learn	Machine learning
🌲 Random Forest	Baseline classifier
⚡ SVM	Main classifier
🧠 MLP	Neural network baseline
📊 Matplotlib	Visualization
🎨 Seaborn	Data visualization
💾 Joblib	Model serialization
🌐 Streamlit	Web application
</div>
🚀 Installation
1. Clone the Repository
git clone https://github.com/AliZain3311/CodeAlpha_Emotion_Recognition.git
2. Navigate to the Project
cd CodeAlpha_Emotion_Recognition
3. Create Virtual Environment
python -m venv venv
4. Activate Environment
Windows
venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
▶️ Run The Application

Start the Streamlit application:

streamlit run app.py
🎧 Command-Line Prediction

The project also provides a command-line prediction script.

python src/predict.py "path\to\audio.wav"

Example:

python src/predict.py ".\data\Audio_Speech_Actors_01-24\Actor_20\03-01-03-01-01-01-20.wav"

Example output:

Predicted Emotion : HAPPY
Confidence         : 48.24%

Top Emotion Probabilities:
happy        : 48.24%
surprised    : 37.47%
angry        :  5.17%
sad          :  3.64%
fearful      :  3.48%
disgust      :  1.48%
neutral       : 0.27%
calm          : 0.26%
🔬 Complete ML Workflow
┌───────────────────────────────┐
│       RAVDESS Dataset         │
│        1,440 Samples          │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Audio Preprocessing     │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│    Advanced Feature Extraction│
│          307 Features         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│   Actor-Based Train/Test Split│
│       19 / 5 Actors           │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Standardization         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Model Benchmarking       │
│ RF • SVM • MLP                │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Feature Selection       │
│        SelectKBest             │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│    SVM Hyperparameter Tuning   │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│    Actor-Aware Validation     │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Final Evaluation        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Saved ML Pipeline       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│     Streamlit Web Application  │
└───────────────────────────────┘
📚 What I Learned

This project provided practical experience with:

🎧 Speech signal processing
🎵 Audio feature engineering
🧠 MFCC extraction
📊 Exploratory data analysis
🤖 Machine learning classification
🔎 Feature selection
⚙️ Hyperparameter optimization
🧪 Cross-validation
🔐 Data leakage prevention
👥 Actor-independent evaluation
📈 Model evaluation
💾 Model serialization
🌐 Streamlit application development
🏗️ End-to-end ML project architecture
⚠️ Limitations

Speech emotion recognition is a difficult machine learning problem.

The current system achieves approximately 51% accuracy on completely unseen actors.

The lower performance on some emotions, particularly Neutral, Sad, and Calm, indicates that these emotional states can be difficult to distinguish from acoustic features alone.

This project prioritizes honest actor-independent evaluation rather than maximizing a random-split accuracy score.

🔮 Future Improvements

Potential improvements include:

🧠 CNN-based spectrogram classification
🔥 Deep learning architectures
🎯 Transfer learning
🎧 Audio data augmentation
🎤 Real-time microphone input
🔄 Temporal models such as LSTM/GRU
🤝 Ensemble learning
📚 Larger and more diverse datasets
☁️ Cloud deployment
📱 Mobile-friendly deployment
📌 Project Status
<div align="center">
✅ Core Project Completed
Component	Status
Dataset Processing	✅
Feature Extraction	✅
Actor-Based Split	✅
Baseline Models	✅
Feature Selection	✅
SVM Optimization	✅
Actor-Aware Evaluation	✅
Model Serialization	✅
CLI Prediction	✅
Streamlit Application	✅
GitHub Repository	✅
</div>
👨‍💻 Author
<div align="center">
Ali Zain

Python • Machine Learning • Audio AI • Streamlit

Built as part of the CodeAlpha Internship.

<br> <a href="https://github.com/AliZain3311"> <img src="https://img.shields.io/badge/GitHub-Ali%20Zain-181717?style=for-the-badge&logo=github" alt="GitHub"> </a> <a href="https://github.com/AliZain3311/CodeAlpha_Emotion_Recognition"> <img src="https://img.shields.io/badge/Project-Repository-8B5CF6?style=for-the-badge&logo=github" alt="Project Repository"> </a> </div>
⭐ Support

If you found this project interesting, consider giving the repository a ⭐ on GitHub.

Your feedback and suggestions are always welcome.

<div align="center">
🎙️ Turning Speech Into Emotion With Machine Learning

Built with Python • Machine Learning • Audio Processing • Streamlit

<br>

⭐ Star the repository if you like the project! ⭐

</div> ```