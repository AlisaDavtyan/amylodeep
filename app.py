import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from prediction import  RollingWindowPredictor
from utils import load_models_and_calibrators

# Page setup
st.set_page_config(page_title="Amyloid Sequence Classifier", layout="wide")

# Custom header and styling
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #F8F9FF !important;
    }
    /* Remove extra top padding so the header hugs the top edge */
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
        margin-bottom: 10px; /* Small gap after the header */
    }
    .custom-header .logo {
        font-size: 32px;
        font-weight: 700;
        color: #1a355e;
    }
    .custom-header .nav-links a {
        margin-left: 25px;
        text-decoration: none;
        font-size: 22px;
        color: #1a355e;
    }
    .custom-header .nav-links a:hover {
        text-decoration: underline;
    }
    #MainMenu, footer, header, .viewerBadge_container__1QSob {
        visibility: hidden !important;
        display: none !important;
    }
    .stFormSubmitButton button {
        background-color: #FFFFFF !important;
        border: 2px solid #00b300 !important;
        width: 150px !important;
        height: 50px !important;
        border-radius: 8px !important;
    }

    .stFormSubmitButton button:hover {
        background-color: #bcf5bc !important;
        border-color: #00b300 !important;
        font-color: #000000
    }

    .stFormSubmitButton button p {
        font-size: 18px !important;
        margin: 0 !important;
    }

    .stFormSubmitButton button div {
        font-size: 18px !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif !important;
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
st.markdown(
    """
    <h5 style='text-align: center; margin-top: 15px; margin-bottom: 20px;'>
        pLM-based model for predicting amyloid propensity from the amino acid sequence.<br/>
        To access full ensemble model follow 
        <a href="https://pypi.org/project/amylodeep/" target="_blank">source</a>.
    </h5>
    """,
    unsafe_allow_html=True
)


# FASTA Parser
def parse_fasta(fasta_text):
    lines = fasta_text.strip().split('\n')
    return ''.join(line.strip() for line in lines if not line.startswith(">"))

# Input layout
left_col, right_col = st.columns([1.2, 2])

with left_col:
    with st.form("sequence_form"):
        # Title above text area
        st.markdown(
            "<div style='font-size:15px; color:#666666; margin-bottom: 0px; line-height: 1;'>Paste amino acid sequence:</div>",
            unsafe_allow_html=True
        )
        sequence_input = st.text_area(" ", key="sequence_input", height=200)

        # Upload section
        st.markdown(
            "<div style='font-size:15px; color:#666666; margin-top: 5px; margin-bottom: 2px; line-height: 1;'>or Upload FASTA file</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(" ", type=["fasta", "fa", "txt"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            sequence_input = parse_fasta(content)


        # Rolling window slider
        window_size = st.slider("Rolling window size:", min_value=3, max_value=30, value=10, step=1, key="window_slider")
        predict_btn = st.form_submit_button("Predict")



# Load predictor
@st.cache_resource(show_spinner=False)
# def get_predictor():
#     model, calibrator= load_models_and_calibrators()
#     return RollingWindowPredictor(model, calibrator)
def get_predictor():
    try:
        model, calibrator = load_models_and_calibrators()
        return RollingWindowPredictor(model=model, calibrator=calibrator)
    except Exception as e:
        st.error("Failed to load predictor.")
        st.exception(e)  # show full traceback in local or cloud logs
        raise e

predictor = get_predictor()

# Prediction
if predict_btn and sequence_input:
    sequence = sequence_input.strip().upper()
    if not sequence.isalpha():
        st.error("Invalid input. Please enter only alphabetic amino acid codes (A-Z).")
    else:
        start_time = time.time()
        result = predictor.rolling_window_prediction(sequence, window_size)
        end_time = time.time()

        with right_col:
            # st.markdown(f"**⏱ Sequence:**")
            st.markdown(f"<h4 style='font-size: 18px;'>{sequence}</h4>", unsafe_allow_html=True)
            # st.markdown("<h5 style='text-align: center;'>Window-wise Amyloidogenicity probability plot using AmyloDee</h5>", unsafe_allow_html=True)

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
            # ax.set_xlim(-1, len(sequence))
            ax.set_xlim(-1, L - window_size + 1)

            if L < 100:
                ax.set_xticks(np.arange(0, L+1, 5))
            else:
                # labels at residues [0, L/5, 2L/5, ..., L]
                step = int(np.ceil(L/5/10) * 10)
                tick_labels = np.arange(0, L+1, step)

                # convert those residue labels to bar x-positions (window starts)
                # last label "L" maps to the last window start at L - window_size
                tick_positions = np.minimum(tick_labels, L - window_size)

                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(t) for t in tick_labels])
            # ax.set_ylim(0, 1)
            # if len(sequence) < 100:
            #     ax.set_xticks(np.arange(0, len(sequence),5))
            # else:
            #     ax.set_xticks(np.arange(0, len(sequence),50))
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_title('Amyloidogenicity probability per window ', fontsize=16)
            st.pyplot(fig)

            st.markdown("<h5 style='text-align: center;'>Position-wise Probabilities</h5>", unsafe_allow_html=True)
            df = pd.DataFrame({
                # "start": [pos for pos, _ in result["position_probs"]],
                "sequence": result["windows"],
                "probability": [p for _, p in result["position_probs"]]
            })


            st.dataframe(df.style.format({"Probability": "{:.3f}"}), use_container_width=True)
