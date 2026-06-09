import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from prediction import RollingWindowPredictor
from utils import load_models_and_calibrators
import textwrap

# Page setup
st.set_page_config(page_title="Amyloid Sequence Classifier", layout="wide")

# Check if we're on the model info page
query_params = st.query_params
active_page = query_params.get("page", "main")
show_model_info = active_page == "model"
show_contact_info = active_page == "contact"

# Custom header and styling
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #F8F9FF !important;
    }
    .block-container {
        padding-top: 0 !important;
    }
    .custom-header {
        background-color: #F8F9FF;
        padding: 20px 20px 10px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Segoe UI', sans-serif;
        border-bottom: 1px solid #1a355e;
        margin-bottom: 10px;
    }
    .custom-header .logo {
        font-size: 32px;
        font-weight: 700;
        color: #1a355e;
        cursor: pointer;
    }
    .custom-header .nav-links a {
        margin-left: 25px;
        text-decoration: none;
        font-size: 22px;
        color: #1a355e;
        cursor: pointer;
    }
    .custom-header .nav-links a:hover {
        text-decoration: underline;
    }
    #MainMenu, footer, header, .viewerBadge_container__1QSob {
        visibility: hidden !important;
        display: none !important;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #FFFFFF !important;
        border: 2px solid #00b300 !important;
        width: 150px !important;
        height: 50px !important;
        border-radius: 8px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #bcf5bc !important;
        border-color: #00b300 !important;
    }
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button div {
        font-size: 20px !important;
        font-weight: normal !important;
        margin: 0 !important;
    }
    div[data-testid="stButton"] > button {
        background-color: #FFFFFF !important;
        border: 2px solid #cc0000 !important;
        width: 150px !important;
        height: 50px !important;
        border-radius: 8px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #ffe5e5 !important;
        border-color: #cc0000 !important;
    }
    .model-info-section {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        margin: 20px auto;
        max-width: 1000px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .model-info-section h1 {
        color: #1a355e;
        font-size: 28px;
        margin-bottom: 20px;
    }
    .model-info-section h2 {
        color: #1a355e;
        font-size: 22px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .model-info-section p {
        font-size: 16px;
        line-height: 1.6;
        color: #333;
    }
    .model-info-section ul {
        font-size: 16px;
        line-height: 1.8;
        color: #333;
    }
    .model-info-section a {
        color: #0066cc;
        text-decoration: none;
    }
    .model-info-section a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="custom-header">
        <a href="?" target="_self" style="text-decoration: none;"><div class="logo">AmyloDeep</div></a>
        <div class="nav-links">
            <a href="?page=model" target="_self">Model</a>
            <a href="?page=contact" target="_self">Contact us</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Show Model Info Page
if show_model_info:
    st.markdown(
        """
        <div class="model-info-section" style="font-size:18px;">
        <h1>🧬 AmyloDeep: pLM-based ensemble model for predicting amyloid propensity from the amino acid sequence</h1>
        <p style="text-align: justify;">
        This tool predicts the amyloid-forming propensity of protein sequences using 
        transformer-based protein embeddings. Amyloids are β-sheet-rich, stable structures that play roles 
        in both critical biological functions (like memory and immunity) and serious diseases such as 
        Alzheimer's, Parkinson's, and Huntington's.
        Amyloids are predominantly β-sheet-rich, stable protein structures that can maintain their presence in the human body
        for multiple years. Amyloid protein aggregates contribute to the development of multiple neurodegenerative diseases, 
        such as Alzheimer's, Parkinson's, and Huntington's, and are involved in different vital functions, such as memory 
        formation and immune system function. Here, we used advanced machine learning and deep learning techniques to predict 
        amyloid propensity from the amino acid sequence. First, we aggregated labeled amino acid sequence data from multiple
        sources, obtaining a roughly balanced dataset of 2366 sequences for binary classification. We leveraged that data to 
        both fine-tune the ESM2 model and to train new models based on protein embeddings from ESM2 and UniRep.
        The predictions from these models were then unified into a single soft voting ensemble model, yielding highly
        robust and accurate results. We further made a tool where users can provide the amino acid sequence and get
        the amyloid formation probabilities of different segments of the input sequence. 
        </p>
        <p>
        AmyloDeep provides reliable predictions of amyloidogenic regions directly from the amino acid sequence.
        </p>
        <h2>🔗 Try It</h2>
        <ul>
        <strong>Package</strong>: the full model is available as a Python package at 
        <a href="https://pypi.org/project/amylodeep/" target="_blank">https://pypi.org/project/amylodeep/</a>
        </ul>
        <h2>📄 Read Full Article</h2>
        <p>
        For detailed information about the methodology and research behind AmyloDeep, read the full article:
        <br/>
        <a href="https://www.biorxiv.org/content/10.1101/2025.09.16.676495v1.full" target="_blank">
        https://www.biorxiv.org/content/10.1101/2025.09.16.676495v1.full
        </a>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# Show Contact Page
if show_contact_info:
    st.markdown(
        """
        <div class="model-info-section" style="font-size:17px;">
        <h1>📬 Contact</h1>
        <p>
        <strong>Authors & Correspondence:</strong><br>
        📧 Alisa Davtyan - <a href="mailto:alisadavtyan7@gmail.com">alisadavtyan7@gmail.com</a><br>
        📧 Anahit Khachatryan - <a href="mailto:xachatryan96an@gmail.com">xachatryan96an@gmail.com</a><br>
        📧 Rafayel Petrosyan - <a href="mailto:rafayel.petrosyan@aua.am">rafayel.petrosyan@aua.am</a>
        </p>
        <h2>📄 Citation</h2>
        <p>
        A. Davtyan, A. Khachatryan, R. Petrosyan "AmyloDeep: pLM-based ensemble model for predicting amyloid propensity from the amino acid sequence", 2025 bioRxiv 
        doi: <a href="https://doi.org/10.1101/2025.09.16.676495" target="_blank">https://doi.org/10.1101/2025.09.16.676495</a>
        </p>
        <br/>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

st.markdown("""
<h5 style='text-align:center; margin-top:15px; margin-bottom:20px;'>
    pLM-based model for predicting amyloid propensity from amino acid sequence.
    <br/> Full ensemble model: 
    <a href="https://pypi.org/project/amylodeep/" target="_blank">source</a>
</h5>
""", unsafe_allow_html=True)

# FASTA Parser
def parse_fasta(fasta_text):
    lines = fasta_text.strip().split('\n')
    sequence = ''.join(line.strip() for line in lines if not line.startswith(">"))
    sequence = ''.join(c for c in sequence if c.isalpha())
    return sequence

# Session state init
if "clean_sequence" not in st.session_state:
    st.session_state.clean_sequence = ""

# Input layout
left_col, right_col = st.columns([1.2, 2])

with left_col:
    with st.form("sequence_form"):
        st.markdown(
            "<div style='font-size:15px; color:#666666; margin-bottom: 0px; line-height: 1;'>Paste amino acid sequence:</div>",
            unsafe_allow_html=True
        )
        sequence_input = st.text_area(" ", key="sequence_input", height=200)

        st.markdown(
            "<div style='font-size:15px; color:#666666; margin-top: 5px; margin-bottom: 2px; line-height: 1;'>or Upload FASTA file</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(" ", type=["fasta", "fa", "txt"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            st.session_state.clean_sequence = parse_fasta(content).upper()

        window_size = st.slider("Rolling window size:", min_value=3, max_value=30, value=10, step=1, key="window_slider")
        predict_btn = st.form_submit_button("Predict")

    # Citation
    st.markdown(
        """
        <div style='font-size:15px; color:#666666; margin-top: 15px; line-height: 1.4;'>
        <strong>Citation:</strong><br>
        <p style="text-align: justify;">
        A. Davtyan, A. Khachatryan, R. Petrosyan "AmyloDeep: pLM-based ensemble model for predicting amyloid propensity from the amino acid sequence", 2025 bioRxiv<br>
        doi: <a href="https://doi.org/10.1101/2025.09.16.676495" target="_blank" style="color:#0066cc;">https://doi.org/10.1101/2025.09.16.676495</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Load predictor
@st.cache_resource(show_spinner=False)
def get_predictor():
    try:
        model, calibrator = load_models_and_calibrators()
        return RollingWindowPredictor(model=model, calibrator=calibrator)
    except Exception as e:
        st.error("Failed to load predictor.")
        st.exception(e)
        raise e

predictor = get_predictor()

# Prediction
if predict_btn:
    raw = sequence_input.strip() if sequence_input.strip() else st.session_state.clean_sequence
    if raw.startswith(">"):
        sequence = parse_fasta(raw).upper()
    else:
        sequence = ''.join(c for c in raw if c.isalpha()).upper()

    if not sequence:
        st.error("No valid amino acid sequence found. Please check your input.")
    else:
        start_time = time.time()
        result = predictor.rolling_window_prediction(sequence, window_size)
        end_time = time.time()

        with right_col:
            wrapped = textwrap.fill(sequence, width=80)
            st.markdown(
                f"<pre style='font-size:14px; white-space:pre-wrap; word-break:break-all; "
                f"background:#f0f0f0; padding:12px; border-radius:6px;'><strong>{wrapped}</strong></pre>",
                unsafe_allow_html=True
            )

            positions, probs = zip(*result['position_probs'])
            x = np.arange(0, len(sequence) - window_size + 1)
            bar_colors = [
                (0, 0, 1, 0.8) if p > 0.8 else (0, 0, 1, 0.6) if p > 0.5 else (0, 0, 1, 0.2) for p in probs
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x, probs, color=bar_colors, width=1, edgecolor="black")
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_xlabel("Residue", fontsize=12)
            L = len(sequence)
            ax.set_xlim(-1, L - window_size + 1)

            if L < 100:
                ax.set_xticks(np.arange(0, L+1, 5))
            else:
                step = int(np.ceil(L/5/10) * 10)
                tick_labels = np.arange(0, L+1, step)
                tick_positions = np.minimum(tick_labels, L - window_size)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(t) for t in tick_labels])

            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_title('Amyloidogenicity probability per window', fontsize=16)
            st.pyplot(fig)

            st.markdown("<h5 style='text-align: center;'>Position-wise Probabilities</h5>", unsafe_allow_html=True)
            df = pd.DataFrame({
                "sequence": result["windows"],
                "probability": [p for _, p in result["position_probs"]]
            })
            st.dataframe(
                df.style.format({"probability": "{:.4f}"})
                  .set_properties(subset=["probability"], **{"text-align": "center"}),
                use_container_width=True
            )


import os

if os.path.exists("OFFLINE.txt"):
    st.set_page_config(page_title="AmyloDeep")
    st.warning("App is currently offline. Check back later.")
    st.stop()