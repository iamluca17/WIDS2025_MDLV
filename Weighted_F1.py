import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

def weighted_f1(y_true, y_pred):
    """
    Computes a weighted F1 score for two binary targets.
    
    For each sample, the two binary targets (assumed to be in two columns) are combined into a single label:
        combined_label = target1 * 2 + target2
    This encoding yields:
        [0, 1, 2, 3] corresponding to:
        [0, 0] -> 0
        [0, 1] -> 1
        [1, 0] -> 2
        [1, 1] -> 3
    Samples with a combined label of 3 (i.e. both targets equal 1) will have their weight doubled.
    
    Parameters
    ----------
    y_true : pandas.DataFrame or np.ndarray, shape (n_samples, 2)
        The true binary target values.
    y_pred : pandas.DataFrame or np.ndarray, shape (n_samples, 2)
        The predicted binary target values.
    
    Returns
    -------
    float
        The weighted F1 score.
    """
    # Convert to numpy arrays if inputs are pandas DataFrames
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
 
    # Combine the two binary targets into one label
    y_true_combined = y_true[:, 0] * 2 + y_true[:, 1]
    y_pred_combined = y_pred[:, 0] * 2 + y_pred[:, 1]
    
    # Create sample weights: weight = 2 if the true combined label is 3, else weight = 1
    sample_weights = np.where(y_true_combined == 3, 2, 1)
    
    # Compute the weighted F1 score using sklearn's f1_score with sample_weight
    score = f1_score(y_true_combined, y_pred_combined, average='weighted', sample_weight=sample_weights)
    
    return score