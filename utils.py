from unirep_model import UniRepClassifier
import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification,logging
import pickle
from ensemble_prediction import EnsembleRollingWindowPredictor  
import wandb
import warnings
import os

os.environ["WANDB_MODE"] = "disabled"
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

@st.cache_resource(show_spinner=False)
def load_models_and_calibrators():
    models = {}
    
    # Initialize wandb API (no run creation)
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    # Model 1: ESM2 150M fine-tuned
    artifact_1 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/final_esm2_150M_checkpoint_100_epochs:v0')
    model_path_1 = artifact_1.download()
    models['esm2_150M'] = AutoModelForSequenceClassification.from_pretrained(model_path_1)
    tokenizer_1 = AutoTokenizer.from_pretrained(model_path_1)
    # Model 2: UniRep classifier
    artifact_2 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/final_UniRepClassifier_4_layers_50_epochs:v0')
    model_path_2 = artifact_2.download()
    models['unirep'] = UniRepClassifier.from_pretrained(model_path_2)
    
    
    # Calibrators
    artifact_3 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/platt_unirep:v0')
    model_path_3 = artifact_3.download()
    calibrator_path = os.path.join(model_path_3, "platt_unirep.pkl")

    calibrators = {}
    with open(calibrator_path, "rb") as f:
        calibrators["platt_unirep"] = pickle.load(f)


    return models, calibrators ,tokenizer_1


def predict_ensemble_rolling(sequence: str, window_size: int = 6):
    """
    Run ensemble prediction with rolling window over a single sequence.
    Returns dictionary with average/max probs and position-wise scores.
    """
    models, calibrators , model_path_1 = load_models_and_calibrators()
    predictor = EnsembleRollingWindowPredictor(models, calibrators,model_path_1)
    return predictor.rolling_window_prediction(sequence, window_size)
