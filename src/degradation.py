import numpy as np
import pandas as pd

def compute_degradation_indicators(reconstruction_errors, window_size=144, threshold=None):
    """
    Computes rolling statistics on the reconstruction error time-series to identify
    gradual sensor degradation and calibration drift.
    
    Parameters:
    -----------
    reconstruction_errors : np.ndarray or pd.Series
        Time-series of sequence reconstruction errors.
    window_size : int
        Rolling window size (default: 144 steps = 24 hours of 10-minute readings).
    threshold : float or None
        Baseline anomaly threshold.
        
    Returns:
    --------
    degradation_df : pd.DataFrame
        DataFrame containing raw errors, rolling mean, rolling std, and drift signals.
    """
    errors_series = pd.Series(reconstruction_errors, name="Reconstruction_Error")
    
    rolling_mean = errors_series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = errors_series.rolling(window=window_size, min_periods=1).std().fillna(0)
    rolling_trend = rolling_mean.diff().fillna(0)

    degradation_df = pd.DataFrame({
        'Error': errors_series,
        'Rolling_Mean': rolling_mean,
        'Rolling_Std': rolling_std,
        'Rolling_Trend': rolling_trend
    })
    
    if threshold is not None:
        # A degradation alert is triggered when the rolling mean error stays persistently high
        degradation_df['Persistent_Degradation_Alert'] = (rolling_mean > (threshold * 0.75)).astype(int)
        
    return degradation_df
