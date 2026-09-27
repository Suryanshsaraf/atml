import numpy as np

def explain_anomaly(sequence_index, X_seq, model, feature_names, threshold=None):
    """
    Computes feature-wise reconstruction error attribution for a specific sequence.
    Calculates exact percentage contribution of each sensor parameter to the detected anomaly.
    
    Parameters:
    -----------
    sequence_index : int
        Index of the sequence to explain.
    X_seq : np.ndarray of shape (n_samples, timesteps, n_features)
        Dataset sequences.
    model : Keras Model
        Trained LSTM Autoencoder.
    feature_names : list of str
        List of feature column names.
    threshold : float or None
        Anomaly decision threshold.
        
    Returns:
    --------
    explanation : dict
        Detailed feature-wise breakdown of the anomaly.
    """
    sample = X_seq[sequence_index : sequence_index + 1]
    reconstruction = model.predict(sample, verbose=0)
    
    # Difference squared: shape (1, timesteps, n_features)
    diff_sq = np.square(sample - reconstruction)
    
    # Average reconstruction error across timesteps for each feature: shape (n_features,)
    feature_errors = np.mean(diff_sq[0], axis=0)
    anomaly_score = float(np.mean(feature_errors))
    
    # Normalized feature contributions (% of total reconstruction error)
    total_error = float(np.sum(feature_errors))
    if total_error > 1e-12:
        contributions = (feature_errors / total_error) * 100.0
    else:
        contributions = np.zeros_like(feature_errors)
        
    top_feature_idx = int(np.argmax(contributions))
    top_feature = feature_names[top_feature_idx]
    top_feature_pct = float(contributions[top_feature_idx])
    
    is_anomaly = None
    if threshold is not None:
        is_anomaly = bool(anomaly_score > threshold)
        
    explanation = {
        'sequence_index': int(sequence_index),
        'anomaly_score': float(anomaly_score),
        'threshold': float(threshold) if threshold is not None else None,
        'is_anomaly': is_anomaly,
        'feature_errors': {f: float(e) for f, e in zip(feature_names, feature_errors)},
        'contributions_percent': {f: float(round(c, 2)) for f, c in zip(feature_names, contributions)},
        'top_feature': top_feature,
        'top_feature_percent': round(top_feature_pct, 2)
    }
    return explanation

def format_explanation_summary(explanation):
    """
    Generates human-readable string summary of the anomaly attribution.
    """
    status = "ANOMALY DETECTED" if explanation.get('is_anomaly') else "NORMAL / SUSPICIOUS"
    lines = [
        f"--- Explanation for Sequence #{explanation['sequence_index']} ({status}) ---",
        f"Anomaly Score: {explanation['anomaly_score']:.5f} (Threshold: {explanation['threshold']})",
        f"Primary Contributor: {explanation['top_feature']} ({explanation['top_feature_percent']}%)",
        "Feature Contributions:"
    ]
    # Sort by contribution descending
    sorted_contrib = sorted(explanation['contributions_percent'].items(), key=lambda x: x[1], reverse=True)
    for feat, pct in sorted_contrib:
        lines.append(f"  * {feat:<20}: {pct:>5.2f}% (MSE: {explanation['feature_errors'][feat]:.5f})")
    return "\n".join(lines)
