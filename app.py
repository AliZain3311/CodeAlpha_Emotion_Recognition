import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import extract_features


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/advanced_emotion_model.joblib"

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

EMOTION_ICONS = {
    "angry": "🔥",
    "calm": "🌿",
    "disgust": "◉",
    "fearful": "◌",
    "happy": "✦",
    "neutral": "—",
    "sad": "◆",
    "surprised": "⚡",
}

EMOTION_COLORS = {
    "angry": "#FB7185",
    "calm": "#2DD4BF",
    "disgust": "#A78BFA",
    "fearful": "#F59E0B",
    "happy": "#FACC15",
    "neutral": "#94A3B8",
    "sad": "#60A5FA",
    "surprised": "#22D3EE",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EmotionAI | Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SMALL HELPER
# ------------------------------------------------------------
# IMPORTANT: every HTML string below is written flush-left
# (no leading indentation on each line). Streamlit's markdown
# renderer follows CommonMark rules, where any block of lines
# indented by 4+ spaces is treated as a literal *code block*
# instead of being parsed as HTML. That is exactly why raw
# "<div class=...>" text was leaking onto the page. Keeping
# every injected HTML block flush-left avoids that permanently.
# ============================================================

def render(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS
# ============================================================

render("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
--bg: #07111F;
--surface: #0B1728;
--card: rgba(15, 29, 48, .78);
--card-strong: rgba(13, 27, 45, .94);
--border: rgba(148, 163, 184, .14);
--text: #F8FAFC;
--muted: #94A3B8;
--primary: #14B8A6;
--cyan: #06B6D4;
--sky: #38BDF8;
}

html, body, [class*="css"] {
font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
background:
radial-gradient(circle at 3% 0%, rgba(20,184,166,.15), transparent 28%),
radial-gradient(circle at 98% 8%, rgba(56,189,248,.12), transparent 25%),
linear-gradient(145deg, #07111F 0%, #091522 48%, #0D1D30 100%);
color: var(--text);
}

[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; }

.block-container { max-width: 1440px; padding: 2rem 2.5rem 4rem; }

[data-testid="stHeader"] { background: rgba(7,17,31,.72); backdrop-filter: blur(18px); }

#MainMenu, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
background: linear-gradient(180deg, #081321 0%, #06101C 100%);
border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #E2E8F0; }

.sidebar-brand { padding: 8px 0 20px; }
.sidebar-logo {
width: 56px; height: 56px; border-radius: 18px;
display: flex; align-items: center; justify-content: center;
font-size: 25px;
background: linear-gradient(135deg, var(--primary), var(--cyan));
box-shadow: 0 14px 35px rgba(20,184,166,.24);
}
.sidebar-title { margin-top: 14px; font-size: 1.48rem; font-weight: 900; letter-spacing: -.035em; color: #fff; }
.sidebar-subtitle { color: #8192A8; font-size: .80rem; line-height: 1.55; margin-top: 3px; }
.sidebar-card {
padding: 15px; margin: 11px 0; border-radius: 18px;
background: rgba(255,255,255,.035); border: 1px solid var(--border);
box-shadow: 0 12px 30px rgba(0,0,0,.14);
}
.sidebar-card-title { color: #5EEAD4; font-size: .70rem; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 9px; }
.sidebar-stat { color: #DCE7F2; font-size: .82rem; margin: 7px 0; }
.sidebar-emotion-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.sidebar-emotion { padding: 7px 8px; border-radius: 10px; background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.06); color: #CBD5E1; font-size: .74rem; display: flex; align-items: center; gap: 6px; }
.sidebar-muted { color: #8192A8; font-size: .73rem; line-height: 1.5; }

/* Hero */
.hero {
position: relative; overflow: hidden;
padding: 46px 48px; border-radius: 30px;
background: linear-gradient(135deg, rgba(20,184,166,.17), rgba(6,182,212,.09) 52%, rgba(56,189,248,.05));
border: 1px solid rgba(94,234,212,.16);
box-shadow: 0 30px 90px rgba(0,0,0,.28);
backdrop-filter: blur(22px);
margin-bottom: 30px;
}
.hero::before, .hero::after { content: ""; position: absolute; border-radius: 50%; pointer-events: none; }
.hero::before { width: 340px; height: 340px; right: -140px; top: -180px; background: radial-gradient(circle, rgba(56,189,248,.18), transparent 68%); }
.hero::after { width: 230px; height: 230px; left: -115px; bottom: -145px; background: radial-gradient(circle, rgba(20,184,166,.12), transparent 68%); }

.hero-kicker {
display: inline-flex; padding: 7px 12px; border-radius: 999px;
background: rgba(20,184,166,.10); border: 1px solid rgba(94,234,212,.18);
color: #5EEAD4; font-size: .70rem; font-weight: 800; letter-spacing: .09em;
}
.hero-title {
position: relative; margin: 18px 0 12px;
font-size: clamp(2.3rem, 5vw, 4.15rem); line-height: 1.02;
font-weight: 900; letter-spacing: -.055em; color: #fff;
}
.hero-gradient {
background: linear-gradient(90deg, #5EEAD4, #38BDF8, #67E8F9);
-webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-description { max-width: 820px; color: #A9B8C9; font-size: 1rem; line-height: 1.75; margin-bottom: 22px; }
.hero-pills { display: flex; flex-wrap: wrap; gap: 9px; }
.hero-pill { padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09); color: #C9D6E5; font-size: .74rem; font-weight: 600; }

/* Sections */
.section-heading { display: flex; align-items: center; gap: 12px; margin: 30px 0 15px; color: #F8FAFC; font-size: 1.28rem; font-weight: 800; letter-spacing: -.02em; }
.section-number {
width: 34px; height: 34px; border-radius: 11px;
display: flex; align-items: center; justify-content: center;
background: linear-gradient(135deg, var(--primary), var(--cyan));
color: #04151A; font-size: .76rem; font-weight: 900;
box-shadow: 0 8px 24px rgba(20,184,166,.22);
}

/* Upload */
.upload-shell { padding: 1px; border-radius: 22px; background: linear-gradient(135deg, rgba(94,234,212,.28), rgba(56,189,248,.18)); margin-bottom: 14px; }
.upload-card { padding: 22px 24px; border-radius: 21px; background: rgba(9,22,37,.92); border: 1px solid rgba(255,255,255,.06); }
.upload-title { font-size: 1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 5px; }
.upload-description { color: #8798AC; font-size: .82rem; line-height: 1.6; }

[data-testid="stFileUploaderDropzone"] {
background: linear-gradient(135deg, rgba(20,184,166,.06), rgba(56,189,248,.05)) !important;
border: 1px dashed rgba(94,234,212,.35) !important;
border-radius: 18px !important;
min-height: 145px !important;
transition: all .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(94,234,212,.70) !important; background: rgba(20,184,166,.08) !important; }
[data-testid="stFileUploaderDropzone"] * { color: #CBD5E1 !important; }

.audio-card { padding: 13px 17px; border-radius: 16px; background: rgba(255,255,255,.035); border: 1px solid var(--border); margin-top: 12px; }
.audio-name { color: #E2E8F0; font-weight: 700; font-size: .84rem; overflow-wrap: anywhere; }

/* Buttons */
.stButton > button {
width: 100%; min-height: 54px;
border: 1px solid rgba(94,234,212,.18) !important;
border-radius: 15px !important;
background: linear-gradient(135deg, #14B8A6, #0891B2) !important;
color: #F8FAFC !important;
font-size: .96rem !important; font-weight: 800 !important;
box-shadow: 0 14px 34px rgba(20,184,166,.18);
transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover { transform: translateY(-2px); filter: brightness(1.08); box-shadow: 0 18px 40px rgba(6,182,212,.24); }

/* Result cards */
.result-card, .probability-card, .metric-card, .tech-box, .table-card {
background: linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.025));
border: 1px solid var(--border);
box-shadow: 0 22px 55px rgba(0,0,0,.20);
backdrop-filter: blur(18px);
}

.result-card { position: relative; overflow: hidden; min-height: 285px; padding: 28px; border-radius: 24px; }
.result-card::after { content: ""; position: absolute; width: 220px; height: 220px; right: -95px; bottom: -115px; border-radius: 50%; background: radial-gradient(circle, rgba(20,184,166,.18), transparent 70%); }
.result-label { color: #8192A8; font-size: .70rem; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; }
.result-icon { font-size: 3rem; margin: 20px 0 5px; }
.result-emotion { font-size: 3rem; font-weight: 950; letter-spacing: -.05em; margin: 0; }
.result-confidence { margin-top: 10px; color: #5EEAD4; font-size: 1rem; font-weight: 800; }
.result-note { margin-top: 20px; color: #8293A8; font-size: .80rem; line-height: 1.6; }

.probability-card { padding: 26px; border-radius: 24px; }
.probability-title { font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 22px; }
.prob-row { margin-bottom: 15px; }
.prob-header { display: flex; justify-content: space-between; margin-bottom: 7px; }
.prob-name { color: #C9D5E3; font-size: .82rem; font-weight: 650; }
.prob-value { color: #F8FAFC; font-size: .81rem; font-weight: 800; }
.prob-track { width: 100%; height: 8px; overflow: hidden; border-radius: 999px; background: rgba(255,255,255,.065); }
.prob-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #14B8A6, #38BDF8); box-shadow: 0 0 16px rgba(56,189,248,.18); }
.prob-row.top .prob-name, .prob-row.top .prob-value { color: #5EEAD4; }

/* Metrics */
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
.metric-card { padding: 18px; border-radius: 18px; }
.metric-icon { color: #5EEAD4; font-size: 1rem; margin-bottom: 10px; }
.metric-value { color: #F8FAFC; font-size: 1.45rem; font-weight: 900; letter-spacing: -.03em; }
.metric-label { color: #718399; font-size: .71rem; margin-top: 5px; }

/* Table */
.table-card { padding: 4px; border-radius: 18px; overflow: hidden; }
.score-table { width: 100%; border-collapse: collapse; color: #CBD5E1; font-size: .83rem; }
.score-table th { text-align: left; color: #8192A8; font-size: .70rem; text-transform: uppercase; letter-spacing: .08em; padding: 13px 16px; background: rgba(255,255,255,.025); }
.score-table td { padding: 12px 16px; border-top: 1px solid rgba(255,255,255,.055); }
.score-table tr.top-row td { color: #5EEAD4; font-weight: 800; background: rgba(20,184,166,.045); }

/* Technical */
.tech-box { padding: 20px; border-radius: 19px; }
.tech-item { padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,.06); }
.tech-item:last-child { border-bottom: none; }
.tech-key { color: #718399; font-size: .70rem; text-transform: uppercase; letter-spacing: .06em; }
.tech-value { color: #DCE6F0; font-size: .87rem; font-weight: 700; margin-top: 3px; }

/* Empty / error */
.empty-state, .error-card { margin-top: 24px; padding: 40px; border-radius: 24px; text-align: center; background: rgba(255,255,255,.025); border: 1px solid var(--border); }
.error-card { text-align: left; border-color: rgba(248,113,113,.20); }

/* Footer */
.footer { margin-top: 55px; padding: 28px; border-radius: 22px; text-align: center; background: linear-gradient(135deg, rgba(20,184,166,.08), rgba(56,189,248,.05)); border: 1px solid var(--border); }
.footer-brand { font-size: 1.02rem; font-weight: 850; color: #F8FAFC; }
.footer-gradient { background: linear-gradient(90deg, #5EEAD4, #38BDF8); -webkit-background-clip: text; background-clip: text; color: transparent; }
.footer-subtitle { margin-top: 7px; color: #718399; font-size: .75rem; }
.footer-badges { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.footer-badge { padding: 6px 10px; border-radius: 999px; background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.07); color: #8B9CAF; font-size: .69rem; }

@media (max-width: 900px) {
.block-container { padding: 1.25rem 1rem 3rem; }
.hero { padding: 30px 25px; border-radius: 24px; }
.metrics-grid { grid-template-columns: repeat(2, 1fr); }
.result-emotion { font-size: 2.5rem; }
}
@media (max-width: 560px) {
.metrics-grid { grid-template-columns: 1fr 1fr; }
.hero-title { font-size: 2.3rem; }
}
</style>
""")


# ============================================================
# FEATURE NAMES
# MUST MATCH THE TRAINING DATASET
# ============================================================

def build_feature_names():
    names = []

    for i in range(40):
        names.append(f"mfcc_mean_{i + 1}")

    for i in range(40):
        names.append(f"mfcc_std_{i + 1}")

    for i in range(40):
        names.append(f"delta_mfcc_mean_{i + 1}")

    for i in range(40):
        names.append(f"delta_mfcc_std_{i + 1}")

    for i in range(40):
        names.append(f"delta2_mfcc_mean_{i + 1}")

    for i in range(40):
        names.append(f"delta2_mfcc_std_{i + 1}")

    for i in range(12):
        names.append(f"chroma_mean_{i + 1}")

    for i in range(12):
        names.append(f"chroma_std_{i + 1}")

    for i in range(7):
        names.append(f"contrast_mean_{i + 1}")

    for i in range(7):
        names.append(f"contrast_std_{i + 1}")

    names.extend([
        "zcr_mean",
        "zcr_std",
        "zcr_min",
        "zcr_max",
        "rms_mean",
        "rms_std",
        "rms_min",
        "rms_max",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_centroid_min",
        "spectral_centroid_max",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_bandwidth_min",
        "spectral_bandwidth_max",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "spectral_rolloff_min",
        "spectral_rolloff_max",
        "pitch_mean",
        "pitch_std",
        "pitch_min",
        "pitch_max",
        "pitch_25",
        "pitch_median",
        "pitch_75",
        "voiced_ratio",
        "tempo",
    ])

    return names


FEATURE_NAMES = build_feature_names()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render("""
<div class="sidebar-brand">
<div class="sidebar-logo">🎙️</div>
<div class="sidebar-title">EmotionAI</div>
<div class="sidebar-subtitle">AI-Powered Speech Emotion Intelligence</div>
</div>
""")

    render("""
<div class="sidebar-card">
<div class="sidebar-card-title">AI Pipeline</div>
<div class="sidebar-stat">🎧 307 Acoustic Features</div>
<div class="sidebar-stat">🎯 SelectKBest → 100 Features</div>
<div class="sidebar-stat">🧠 RBF Support Vector Machine</div>
<div class="sidebar-stat">📊 Actor-Aware Evaluation</div>
</div>
""")

    emotion_html = "".join(
        f'<div class="sidebar-emotion"><span style="color:{EMOTION_COLORS[e]};">{EMOTION_ICONS[e]}</span>{e.title()}</div>'
        for e in EMOTIONS
    )

    render(f"""
<div class="sidebar-card">
<div class="sidebar-card-title">Supported Emotions</div>
<div class="sidebar-emotion-grid">
{emotion_html}
</div>
</div>
""")

    render("""
<div class="sidebar-card">
<div class="sidebar-card-title">Project</div>
<div class="sidebar-muted">RAVDESS Speech Emotion Dataset</div>
<div class="sidebar-muted" style="margin-top:7px;">Machine Learning · Acoustic Feature Engineering · Streamlit</div>
<div class="sidebar-muted" style="margin-top:9px;">Built for the CodeAlpha Emotion Recognition project.</div>
</div>
""")


# ============================================================
# HERO
# ============================================================

render("""
<div class="hero">
<div class="hero-kicker">✦ AI-POWERED AUDIO INTELLIGENCE</div>
<div class="hero-title">Understand the <span class="hero-gradient">emotion behind speech.</span></div>
<div class="hero-description">Upload a speech recording and let our trained machine-learning pipeline analyze acoustic characteristics to estimate the speaker's emotional state.</div>
<div class="hero-pills">
<div class="hero-pill">🎧 Audio Analysis</div>
<div class="hero-pill">🧠 SVM Intelligence</div>
<div class="hero-pill">🎯 8 Emotion Classes</div>
<div class="hero-pill">📈 Probability Insights</div>
</div>
</div>
""")


# ============================================================
# UPLOAD
# ============================================================

render("""
<div class="section-heading">
<div class="section-number">01</div>
Upload your recording
</div>
""")

render("""
<div class="upload-shell">
<div class="upload-card">
<div class="upload-title">🎧 Select a speech recording</div>
<div class="upload-description">Drop your audio file below or browse your device. Supported formats: WAV, MP3, OGG, FLAC and M4A. WAV recordings similar to RAVDESS provide the most consistent results.</div>
</div>
</div>
""")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    label_visibility="collapsed",
    help="For the most consistent results, use a speech recording similar to the RAVDESS dataset.",
)


if uploaded_file is not None:

    render(f"""
<div class="audio-card">
<div class="audio-name">🎵 {uploaded_file.name}</div>
</div>
""")

    st.audio(
        uploaded_file,
        format=uploaded_file.type,
    )

    render("""
<div class="section-heading">
<div class="section-number">02</div>
Analyze the recording
</div>
""")

    analyze = st.button(
        "✦  Analyze Emotion",
        type="primary",
        width="stretch",
    )

    if analyze:

        temp_path = None

        try:

            # ------------------------------------------------
            # Save upload temporarily
            # ------------------------------------------------

            suffix = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            with st.spinner(
                "Loading AI model..."
            ):

                model = load_model()


            # ------------------------------------------------
            # Extract features
            # ------------------------------------------------

            with st.spinner(
                "Extracting acoustic features..."
            ):

                features = extract_features(
                    temp_path
                )

            if len(features) != 307:

                st.error(
                    f"Feature extraction produced {len(features)} "
                    "features. The trained model requires exactly 307."
                )

                st.stop()


            # ------------------------------------------------
            # Create DataFrame with training feature names
            # ------------------------------------------------

            X = pd.DataFrame(
                [features],
                columns=FEATURE_NAMES,
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with st.spinner(
                "Analyzing emotional characteristics..."
            ):

                prediction = model.predict(X)[0]

                probabilities = model.predict_proba(X)[0]

                classes = model.classes_

                probability_dict = dict(
                    zip(
                        classes,
                        probabilities,
                    )
                )


            # ------------------------------------------------
            # Sort probabilities
            # ------------------------------------------------

            probability_dict = dict(
                sorted(
                    probability_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            confidence = (
                probability_dict[prediction] * 100
            )


            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            render("""
<div class="section-heading">
<div class="section-number">03</div>
Emotion Intelligence
</div>
""")

            icon = EMOTION_ICONS.get(prediction, "🎙️")
            emotion_color = EMOTION_COLORS.get(prediction, "#5EEAD4")

            col1, col2 = st.columns([0.85, 1.5], gap="medium")

            with col1:
                render(f"""
<div class="result-card">
<div class="result-label">Predicted Emotion</div>
<div class="result-icon">{icon}</div>
<div class="result-emotion" style="color:{emotion_color};">{prediction.upper()}</div>
<div class="result-confidence">{confidence:.2f}% Confidence</div>
<div class="result-note">The model identified <strong>{prediction.title()}</strong> as the highest-probability emotional class for this recording.</div>
</div>
""")

            with col2:
                probability_html = ""

                for index, (emotion, probability) in enumerate(
                    probability_dict.items()
                ):
                    percentage = probability * 100
                    row_class = "prob-row top" if index == 0 else "prob-row"
                    emotion_icon = EMOTION_ICONS.get(emotion, "•")

                    probability_html += (
                        f'<div class="{row_class}">'
                        f'<div class="prob-header">'
                        f'<div class="prob-name">{emotion_icon}&nbsp;&nbsp;{emotion.title()}</div>'
                        f'<div class="prob-value">{percentage:.2f}%</div>'
                        f'</div>'
                        f'<div class="prob-track">'
                        f'<div class="prob-fill" style="width:{percentage:.2f}%;"></div>'
                        f'</div>'
                        f'</div>'
                    )

                render(f"""
<div class="probability-card">
<div class="probability-title">Emotion Probability Profile</div>
{probability_html}
</div>
""")

            # Quick analytics
            render(f"""
<div class="metrics-grid">
<div class="metric-card">
<div class="metric-icon">◈</div>
<div class="metric-value">{len(features)}</div>
<div class="metric-label">Acoustic Features</div>
</div>
<div class="metric-card">
<div class="metric-icon">◇</div>
<div class="metric-value">100</div>
<div class="metric-label">Selected Features</div>
</div>
<div class="metric-card">
<div class="metric-icon">◎</div>
<div class="metric-value">{len(probability_dict)}</div>
<div class="metric-label">Emotion Classes</div>
</div>
<div class="metric-card">
<div class="metric-icon">✦</div>
<div class="metric-value">{confidence:.1f}%</div>
<div class="metric-label">Top Confidence</div>
</div>
</div>
""")

            # ------------------------------------------------
            # Detailed probability table
            # ------------------------------------------------

            render("""
<div class="section-heading">
<div class="section-number">04</div>
Detailed Probability Scores
</div>
""")

            table_rows = ""

            for index, (emotion, probability) in enumerate(
                probability_dict.items()
            ):
                percentage = probability * 100
                row_class = "top-row" if index == 0 else ""
                table_rows += (
                    f'<tr class="{row_class}">'
                    f'<td>{emotion.title()}</td>'
                    f'<td>{percentage:.2f}%</td>'
                    f'</tr>'
                )

            render(f"""
<div class="table-card">
<table class="score-table">
<thead>
<tr><th>Emotion</th><th>Probability</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
""")

            # ------------------------------------------------
            # Technical information
            # ------------------------------------------------

            render("""
<div class="section-heading">
<div class="section-number">05</div>
Model Information
</div>
""")

            render(f"""
<div class="tech-box">
<div class="tech-item">
<div class="tech-key">Input File</div>
<div class="tech-value">{uploaded_file.name}</div>
</div>
<div class="tech-item">
<div class="tech-key">Feature Extraction</div>
<div class="tech-value">307 Acoustic Features</div>
</div>
<div class="tech-item">
<div class="tech-key">Feature Selection</div>
<div class="tech-value">SelectKBest · 100 Features</div>
</div>
<div class="tech-item">
<div class="tech-key">Classifier</div>
<div class="tech-value">RBF Support Vector Machine</div>
</div>
<div class="tech-item">
<div class="tech-key">Preprocessing</div>
<div class="tech-value">StandardScaler</div>
</div>
<div class="tech-item">
<div class="tech-key">Dataset</div>
<div class="tech-value">RAVDESS Speech Emotion Dataset</div>
</div>
</div>
""")

        except Exception as error:

            render("""
<div class="error-card">
<div class="tech-key">Analysis Error</div>
<div class="tech-value" style="margin-top:8px;">Analysis could not be completed.</div>
<div class="sidebar-muted" style="margin-top:8px;">Please try another audio recording. If the problem continues, verify that the trained model and feature extraction pipeline are compatible.</div>
</div>
""")

            with st.expander("Technical error details"):
                st.exception(error)

        finally:

            if temp_path and os.path.exists(
                temp_path
            ):
                os.remove(temp_path)


else:

    render("""
<div class="empty-state">
<div style="font-size:3rem;margin-bottom:12px;">🎙️</div>
<div style="color:#F8FAFC;font-size:1.25rem;font-weight:800;">Your emotion analysis starts here</div>
<div style="color:#718399;margin-top:8px;font-size:.86rem;">Upload a speech recording above to discover the emotional characteristics hidden within the voice.</div>
</div>
""")


# ============================================================
# FOOTER
# ============================================================

render("""
<div class="footer">
<div class="footer-brand"><span class="footer-gradient">EmotionAI</span> &nbsp;·&nbsp; Speech Emotion Intelligence</div>
<div class="footer-subtitle">RAVDESS • Machine Learning • Acoustic Feature Engineering</div>
<div class="footer-badges">
<div class="footer-badge">Python</div>
<div class="footer-badge">Librosa</div>
<div class="footer-badge">Scikit-learn</div>
<div class="footer-badge">RBF SVM</div>
<div class="footer-badge">Streamlit</div>
</div>
</div>
""")