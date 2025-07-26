from unirep_model import UniRepClassifier
import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification,logging
import pickle
from prediction import  RollingWindowPredictor
import wandb
import warnings
import os

os.environ["WANDB_MODE"] = "disabled"
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

@st.cache_resource(show_spinner=False)
def load_models_and_calibrators():
    # models = {}
    
    # Initialize wandb API 
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    # Model: UniRep classifier
    artifact_2 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/final_UniRepClassifier_4_layers_50_epochs:v0')
    model_path_2 = artifact_2.download()
    model = UniRepClassifier.from_pretrained(model_path_2)
    
    # Calibrators
    artifact_3 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/platt_unirep:v0')
    model_path_3 = artifact_3.download()
    calibrator_path = os.path.join(model_path_3, "platt_unirep.pkl")

    # calibrators = {}
    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)


    return model, calibrator


def predict_ensemble_rolling(sequence: str, window_size: int = 6):
    """
    Run ensemble prediction with rolling window over a single sequence.
    Returns dictionary with average/max probs and position-wise scores.
    """
    model, calibrator = load_models_and_calibrators()
    predictor = RollingWindowPredictor(model, calibrator)
    return predictor.rolling_window_prediction(sequence, window_size)
