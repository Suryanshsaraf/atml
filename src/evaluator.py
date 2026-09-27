import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

def compute_reconstruction_error(model, X_seq, batch_size=512):
    """
    Computes both sequence-level MSE and feature-wise MSE for time-series sequences.
    
    Parameters:
    -----------
    model : Keras Model
        Trained LSTM Autoencoder.
    X_seq : np.ndarray of shape (n_samples, timesteps, n_features)
        Input sequences.
    batch_size : int
        Inference batch size.
        
    Returns:
    --------
    seq_errors : np.ndarray of shape (n_samples,)
        Mean Squared Error across timesteps and features for each sequence.
    feature_errors : np.ndarray of shape (n_samples, n_features)
        Mean Squared Error averaged across timesteps for each individual feature.
    X_reconstructed : np.ndarray of shape (n_samples, timesteps, n_features)
        Reconstructed sequences.
    """
    X_reconstructed = model.predict(X_seq, batch_size=batch_size, verbose=0)
    # Difference squared: (n_samples, timesteps, n_features)
    diff_sq = np.square(X_seq - X_reconstructed)
    
    # Sequence-level error: mean across (timesteps, features)
    seq_errors = np.mean(diff_sq, axis=(1, 2))
    
    # Feature-wise error: mean across timesteps
    feature_errors = np.mean(diff_sq, axis=1)
    
    return seq_errors, feature_errors, X_reconstructed

def select_threshold(normal_errors, percentile=95.0, method="percentile"):
    """
    Determines an anomaly threshold based on normal training/validation errors.
    
    Methods:
    --------
    'percentile' : Selects the specified percentile (e.g., 95th or 99th percentile).
    'zscore'     : mean + 3 * std of normal reconstruction error.
    """
    if method == "percentile":
        threshold = float(np.percentile(normal_errors, percentile))
    elif method == "zscore":
        threshold = float(np.mean(normal_errors) + 3.0 * np.std(normal_errors))
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return threshold

def evaluate_predictions(y_true, y_pred, anomaly_scores=None):
    """
    Calculates Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    auc = None
    if anomaly_scores is not None:
        try:
            auc = float(roc_auc_score(y_true, anomaly_scores))
        except Exception:
            auc = None
            
    return {
        'precision': float(p),
        'recall': float(r),
        'f1_score': float(f1),
        'roc_auc': auc,
        'confusion_matrix': cm
    }

def build_comparison_table(results_dict):
    """
    Formats model evaluation dictionaries into a clean comparison DataFrame.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            'Model': model_name,
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1-Score': metrics.get('f1_score', 0.0),
            'ROC-AUC': metrics.get('roc_auc', None)
        })
    df_comp = pd.DataFrame(rows)
    return df_comp
