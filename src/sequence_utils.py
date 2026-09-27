import numpy as np

def create_sequences(X, y=None, window_size=12, label_mode='any', dtype=np.float32):
    """
    Converts 2D time-series feature matrix (samples, features) into 3D sliding
    temporal windows of shape (n_sequences, window_size, n_features).
    
    Parameters:
    -----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    y : np.ndarray or None
        1D binary array of labels (n_samples,).
    window_size : int
        Number of consecutive observations per sequence (default: 12, ~2 hours).
    label_mode : str
        'any'  : Sequence is anomalous if ANY observation in the window is anomalous.
        'last' : Sequence is anomalous if the LAST observation in the window is anomalous.
    dtype : np.dtype
        Floating point data type for sequences (default: np.float32).
        
    Returns:
    --------
    X_seq : np.ndarray of shape (n_samples - window_size + 1, window_size, n_features)
    y_seq : np.ndarray of shape (n_samples - window_size + 1,) if y is provided, else None.
    """
    n_samples, n_features = X.shape
    if n_samples < window_size:
        raise ValueError(f"Sample size ({n_samples}) must be greater than or equal to window_size ({window_size})")

    n_sequences = n_samples - window_size + 1
    
    # Pre-allocate contiguous array to prevent memory fragmentation
    X_seq = np.empty((n_sequences, window_size, n_features), dtype=dtype)
    for i in range(window_size):
        X_seq[:, i, :] = X[i : i + n_sequences, :]

    if y is not None:
        y = np.asarray(y)
        if label_mode == 'any':
            y_windows = np.empty((n_sequences, window_size), dtype=y.dtype)
            for i in range(window_size):
                y_windows[:, i] = y[i : i + n_sequences]
            y_seq = (np.sum(y_windows == 1, axis=1) > 0).astype(int)
        elif label_mode == 'last':
            y_seq = y[window_size - 1 :].astype(int)
        else:
            raise ValueError(f"Unknown label_mode '{label_mode}'. Supported modes: 'any', 'last'.")
        return X_seq, y_seq

    return X_seq
