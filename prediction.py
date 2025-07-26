import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import jax_unirep
import pickle

class RollingWindowPredictor:
    def __init__(self, model, calibrator=None):
        """
        Initialize the UniRep predictor with optional calibration.

        Args:
            model: Trained PyTorch model that accepts UniRep embeddings
            calibrator: Optional sklearn-like calibrator for probability adjustment
        """
        self.model = model
        self.calibrator = calibrator 


    def predict(self, sequences):
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
            outputs = self.model(embeddings=embeddings)
            probs = F.softmax(outputs['logits'], dim=1)[:, 1]

        probs_np = probs.numpy()

        
        if self.calibrator is not None:
            probs_np = self.calibrator.predict_proba(probs_np.reshape(-1, 1))[:, 1]

        return probs_np
    

    def rolling_window_prediction(self, sequence, window_size):
        """
        Predict amyloid probability for an entire sequence using rolling window approach.
        The window slides one position at a time across the sequence.

        Args:
            sequence: Single protein sequence string
            window_size: Size of the sliding window

        Returns:
            dict containing:
                - 'windows': List of window sequences
                - 'position_probs': List of (position, probability) tuples
                - 'window_probs': List of (window_sequence, probability) tuples
                - 'avg_probability': Average probability across all windows
                - 'max_probability': Maximum probability across all windows
                - 'sequence_length': Length of the input sequence
                - 'num_windows': Number of windows
        """
        sequence_length = len(sequence)

        if sequence_length < window_size:
            # If sequence is shorter than window, predict on the entire sequence
            prob = self.predict([sequence])[0]
            return {
                'windows': [sequence], 
                'position_probs': [(0, prob)],
                'window_probs': [(sequence, prob)], 
                'avg_probability': prob,
                'max_probability': prob,
                'sequence_length': sequence_length,
                'num_windows': 1  
            }

    
        windows = []
        positions = []

        for i in range(sequence_length - window_size + 1):
            window = sequence[i:i + window_size]
            windows.append(window)
            positions.append(i)

        # Predict on all windows
        window_probs = self.predict(windows)

        # Combine results
        position_probs = list(zip(positions, window_probs))
        avg_probability = np.mean(window_probs)
        max_probability = np.max(window_probs)

        return {
                'windows': windows,
                'position_probs': position_probs,
                'window_probs': list(zip(windows, window_probs)),  
                'avg_probability': avg_probability,
                'max_probability': max_probability,
                'sequence_length': sequence_length,
                'num_windows': len(windows)
            }