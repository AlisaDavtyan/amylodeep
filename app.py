import streamlit as st
from ensemble_prediction import EnsembleRollingWindowPredictor
from utils import load_models_and_calibrators
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
# Page setup
st.set_page_config(page_title="Amyloid Sequence Classifier", layout="wide")
st.markdown(
    """
    <style>
        /* Reduce padding around the content block */
        .main .block-container {
            padding-top: 0px;
            padding-right: 30px;
            padding-left: 30px;
            padding-bottom: 0px;
        }

        /* Optional: allow full width if layout feels narrow */
        .main .block-container {
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle using HTML
st.markdown("<h1 style='text-align: center; font-size: 3em;'>AmyloDeep</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center;'>Prediction of amyloid propensity from the amino acid sequences using deep learning</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Deep learning-based ensemble model for predicting amyloid propensity from the amino acid sequence</h3>",unsafe_allow_html=True)


# Input form
left_col, right_col = st.columns([1.2, 2])

with left_col:
    with st.form("sequence_form"):
        sequence_input = st.text_area("Paste your protein sequence:", height=200)
        window_size = st.slider("Rolling window size:", min_value=3, max_value=30, value=6, step=1)
        submit = st.form_submit_button("Predict")

# Load models and predictor (cached to avoid reloading)
@st.cache_resource(show_spinner=False)
def get_predictor():
    models, calibrators = load_models_and_calibrators()
    return EnsembleRollingWindowPredictor(models, calibrators)

predictor = get_predictor()
start_time = time.time()
# Prediction logic

if submit:
    sequence = sequence_input.strip().upper()
    if not sequence.isalpha():
        st.error("Invalid input. Please enter only alphabetic amino acid codes (A-Z).")
    elif len(sequence) < window_size:
        st.warning(f"Sequence is shorter than window size ({window_size}). Running full-sequence prediction.")
        result = predictor.rolling_window_prediction(sequence, window_size)
    else:
        result = predictor.rolling_window_prediction(sequence, window_size)
    end_time = time.time()
    total = end_time - start_time

    with right_col:
        st.markdown(f"**⏱ Prediction time:** `{total:.2f} seconds`")
        # st.subheader("📊 Prediction Summary")
        # st.markdown(f"**Sequence length:** {result['sequence_length']}  , **Number of windows:** {result['num_windows']} , **Maximum probability:** `{result['max_probability']:.4f}`")
        # st.markdown(f"**Number of windows:** {result['num_windows']}")
        # st.markdown(f"**Maximum probability:** `{result['max_probability']:.4f}`")
        st.subheader(f"{sequence}")
        st.markdown("<h5 style='text-align: center;'>Position-wise Probability Plot</h5>",unsafe_allow_html=True)
        positions, probs = zip(*result['position_probs'])
        fig, ax = plt.subplots(figsize=(10, 5))
        seq_length = len(sequence)
        x = np.arange(0, seq_length - window_size+1)
        bar_colors = []
        for prob in probs:
            if prob > 0.8:
                bar_colors.append((0, 0, 1, 0.8))
            elif prob > 0.5:
                bar_colors.append((0, 0, 1, 0.6))
            else:
                bar_colors.append((0, 0, 1, 0.2))
        ax.bar(x, probs, color=bar_colors, width=1, edgecolor="black")
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_xlabel("Residue", fontsize=12)
        ax.set_xlim(-1, seq_length)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, seq_length + 1, min(10,len(sequence))))
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title('AmyloDeep ', fontsize=18)
        st.pyplot(fig)

    
        st.markdown("<h5 style='text-align: center;'>Position-wise Probabilities</h5>",unsafe_allow_html=True)
        df = pd.DataFrame(result['position_probs'], columns=["Start Position", "Probability"])
        st.dataframe(df.style.format({"Probability": "{:.4f}"}), use_container_width=True)
