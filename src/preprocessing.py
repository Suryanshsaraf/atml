import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_MAPPING = {
    'T (degC)': 'Temperature',
    'p (mbar)': 'Pressure',
    'rh (%)': 'Humidity',
    'Tdew (degC)': 'DewPoint',
    'wv (m/s)': 'WindSpeed'
}

CORE_FEATURE_NAMES = list(FEATURE_MAPPING.values())
FEATURE_COLS = [
    'Temperature', 'Pressure', 'Humidity', 'DewPoint', 'WindSpeed',
    'DewPoint_Spread', 'Temp_Delta', 'Pressure_Delta', 'Hour_Sin', 'Hour_Cos'
]

def preprocess_weather_data(df_raw):
    """
    Parses timestamps, sorts chronologically, maps core features, and performs
    feature engineering (Dew Point Spread, Deltas, Cyclical Time Encodings).
    """
    df = df_raw.copy()
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('Date Time').reset_index(drop=True)

    df_selected = df[['Date Time'] + list(FEATURE_MAPPING.keys())].rename(columns=FEATURE_MAPPING)

    # Physical constraint: Dew point spread
    df_selected['DewPoint_Spread'] = df_selected['Temperature'] - df_selected['DewPoint']
    
    # Temporal rate of change (deltas per 10-minute step)
    df_selected['Temp_Delta'] = df_selected['Temperature'].diff().fillna(0)
    df_selected['Pressure_Delta'] = df_selected['Pressure'].diff().fillna(0)

    # Diurnal cyclical time encodings
    hours = df_selected['Date Time'].dt.hour + df_selected['Date Time'].dt.minute / 60.0
    df_selected['Hour_Sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_selected['Hour_Cos'] = np.cos(2 * np.pi * hours / 24.0)

    df_selected = df_selected.dropna().reset_index(drop=True)
    return df_selected, FEATURE_COLS

def chronological_split(df_selected, train_ratio=0.80):
    """
    Splits the time-series chronologically into clean training and test sets.
    Preserves strict temporal ordering without data leakage.
    """
    split_idx = int(len(df_selected) * train_ratio)
    train_df = df_selected.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_selected.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df

def fit_and_scale(train_df, test_df, feature_cols=FEATURE_COLS):
    """
    Fits StandardScaler strictly on the clean training set to prevent data leakage,
    then transforms both training and testing feature sets.
    """
    scaler = StandardScaler()
    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    return X_train_scaled, X_test_scaled, scaler
