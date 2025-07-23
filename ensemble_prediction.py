import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import jax_unirep
import pickle


class EnsembleRollingWindowPredictor:
    def __init__(self, models_dict, calibrators_dict=None):
        """
        Initialize the ensemble predictor with all 5 models and calibrators.

        Args:
            models_dict: Dictionary containing all 5 models with keys:
                'esm2_150M', 'unirep', 'esm2_650M', 'svm', 'xgboost'
            calibrators_dict: Dictionary containing calibrators where applicable
        """
        self.models = models_dict
        self.calibrators = calibrators_dict or {}

        # Initialize tokenizers
        self.tokenizer_1 = AutoTokenizer.from_pretrained("../models/final_esm2_150M_checkpoint_100_epochs")


    def _predict_model_1(self, sequences):
        """ESM2 150M fine-tuned model prediction"""
        def tokenize_function(sequences):
            return self.tokenizer_1(sequences, padding="max_length", truncation=True, max_length=128)

        encodings = tokenize_function(sequences)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        with torch.no_grad():
            outputs = self.models['esm2_150M'](input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1]

        return probs.numpy()

    def _predict_model_2(self, sequences):
        """UniRep model prediction"""
        def unirep_tokenize_function(sequences):
            h_final, c_final, h_avg = jax_unirep.get_reps(sequences)
            return {
                "embeddings": h_final,
                "avg_hidden": h_avg,
                "cell_state": c_final
            }

        encodings = unirep_tokenize_function(sequences)
        embeddings = torch.tensor(encodings["embeddings"], dtype=torch.float32)

        with torch.no_grad():
            outputs = self.models['unirep'](embeddings=embeddings)
            probs = F.softmax(outputs['logits'], dim=1)[:, 1]

        probs_np = probs.numpy()

        # Apply calibration if available
        if '../models/platt_unirep' in self.calibrators:
            probs_np = self.calibrators['../models/platt_unirep'].predict_proba(probs_np.reshape(-1, 1))[:, 1]

        return probs_np
    


    def predict_ensemble(self, sequences):
        """
        Predict ensemble probabilities for a list of sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            numpy array of ensemble probabilities
        """
        # Get predictions from all models
        probs_1 = self._predict_model_1(sequences)  # ESM2 150M - NO calibration
        probs_2 = self._predict_model_2(sequences)  # UniRep - WITH calibration (platt_unirep)
    

        # Combine probabilities (matching your original mixed_probs_list order)
        mixed_probs_list = [probs_1, probs_2]

        # Compute average probabilities
        avg_probs = np.mean(mixed_probs_list, axis=0)

        return avg_probs

    def rolling_window_prediction(self, sequence, window_size):
        """
        Predict amyloid probability for an entire sequence using rolling window approach.
        The window slides one position at a time across the sequence.

        Args:
            sequence: Single protein sequence string
            window_size: Size of the sliding window

        Returns:
            dict containing:
                - 'position_probs': List of (position, probability) tuples
                - 'avg_probability': Average probability across all windows
                - 'max_probability': Maximum probability across all windows
                - 'sequence_length': Length of the input sequence
        """
        sequence_length = len(sequence)

        if sequence_length < window_size:
            # If sequence is shorter than window, predict on the entire sequence
            prob = self.predict_ensemble([sequence])[0]
            return {
                'position_probs': [(0, prob)],
                'avg_probability': prob,
                'max_probability': prob,
                'sequence_length': sequence_length
            }

        # Generate windows - slide one position at a time
        windows = []
        positions = []

        for i in range(sequence_length - window_size + 1):
            window = sequence[i:i + window_size]
            windows.append(window)
            positions.append(i)

        # Predict on all windows
        window_probs = self.predict_ensemble(windows)

        # Combine results
        position_probs = list(zip(positions, window_probs))
        avg_probability = np.mean(window_probs)
        max_probability = np.max(window_probs)

        return {
            'position_probs': position_probs,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'sequence_length': sequence_length,
            'num_windows': len(windows)
        }


