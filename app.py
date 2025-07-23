import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from ensemble_prediction import EnsembleRollingWindowPredictor
from utils import load_models_and_calibrators

# Page setup
st.set_page_config(page_title="Amyloid Sequence Classifier", layout="wide")

# Custom header and styling
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #F8F9FF !important;
    }
    .custom-header {
        background-color: #F8F9FF;
        padding: 20px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .custom-header .logo {
        font-size: 26px;
        font-weight: 700;
        color: #1a355e;
    }
    .custom-header .nav-links a {
        margin-left: 40px;
        text-decoration: none;
        font-size: 17px;
        color: #1a355e;
    }
    .custom-header .nav-links a:hover {
        text-decoration: underline;
    }
    #MainMenu, footer, header, .viewerBadge_container__1QSob {
        visibility: hidden !important;
        display: none !important;
    }
    </style>
    <div class="custom-header">
        <div class="logo">AmyloDeep</div>
        <div class="nav-links">
            <a href="#model">Model</a>
            <a href="#datasource">Datasource</a>
            <a href="#researchers">Researchers</a>
            <a href="#contact">Contact</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Intro
st.markdown("<h5 style='text-align: center;'>Deep learning-based ensemble model for predicting amyloid propensity from the amino acid sequence</h5>", unsafe_allow_html=True)

# FASTA Parser
def parse_fasta(fasta_text):
    lines = fasta_text.strip().split('\n')
    return ''.join(line.strip() for line in lines if not line.startswith(">"))

# Input layout
left_col, right_col = st.columns([1.2, 2])

with left_col:
    with st.form("sequence_form"):
        sequence_input = st.text_area("Paste your protein sequence:", height=200)

        st.markdown("**or Upload FASTA file**")
        uploaded_file = st.file_uploader(" ", type=["fasta", "fa", "txt"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            sequence_input = parse_fasta(content)

        window_size = st.slider("Rolling window size:", min_value=3, max_value=30, value=6, step=1, key="window_slider")
        submit = st.form_submit_button("Predict")

# Load predictor
@st.cache_resource(show_spinner=False)
def get_predictor():
    models, calibrators, esm2_path = load_models_and_calibrators()
    return EnsembleRollingWindowPredictor(models, calibrators, esm2_150M_path=esm2_path)

predictor = get_predictor()

# Prediction
if submit and sequence_input:
    sequence = sequence_input.strip().upper()
    if not sequence.isalpha():
        st.error("Invalid input. Please enter only alphabetic amino acid codes (A-Z).")
    else:
        start_time = time.time()
        result = predictor.rolling_window_prediction(sequence, window_size)
        end_time = time.time()

        with right_col:
            st.markdown(f"**⏱ Prediction time:** `{end_time - start_time:.2f} seconds`")
            st.subheader(f"{sequence}")
            st.markdown("<h5 style='text-align: center;'>Position-wise Probability Plot</h5>", unsafe_allow_html=True)

            positions, probs = zip(*result['position_probs'])
            x = np.arange(0, len(sequence) - window_size + 1)
            bar_colors = [
                (0, 0, 1, 0.8) if p > 0.8 else (0, 0, 1, 0.6) if p > 0.5 else (0, 0, 1, 0.2) for p in probs
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x, probs, color=bar_colors, width=1, edgecolor="black")
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_xlabel("Residue", fontsize=12)
            ax.set_xlim(-1, len(sequence))
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(0, len(sequence) + 1, min(10, len(sequence))))
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_title('AmyloDeep', fontsize=18)
            st.pyplot(fig)

            st.markdown("<h5 style='text-align: center;'>Position-wise Probabilities</h5>", unsafe_allow_html=True)
            df = pd.DataFrame({
                "Subsequence": result["windows"],
                "Probability": probs
            })
            st.dataframe(df.style.format({"Probability": "{:.3f}"}), use_container_width=True)
