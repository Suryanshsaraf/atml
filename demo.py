import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_loader import load_data
from src.preprocessing import preprocess_weather_data, FEATURE_COLS
from src.xai import explain_anomaly

def run_viva_demo():
    print("=" * 70)
    print("  AUTOMATIC WEATHER STATION (AWS) ANOMALY DETECTION — VIVA DEMO ")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "lstm_autoencoder.keras")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("\n[ERROR] Trained model or scaler not found in 'models/' directory!")
        print("Please run 'python train_and_evaluate.py' first to generate and save models.")
        return

    # 1. Load artifacts
    print("\n[1/3] Loading Saved Model & Preprocessing Artifacts...")
    lstm_ae = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    threshold = metadata["threshold"]
    window_size = metadata.get("window_size", 12)
    feature_cols = metadata.get("feature_cols", FEATURE_COLS)

    print(f"  * Model Architecture: LSTM Autoencoder (12-step temporal window)")
    print(f"  * Anomaly Threshold:  {threshold:.5f} (95th percentile of normal data)")

    # 2. Ingest clean sample slice
    print("\n[2/3] Extracting Sample Historical Weather Window (2 Hours of Readings)...")
    df_raw = load_data()
    df_clean, _ = preprocess_weather_data(df_raw.iloc[50000:50100])

    # Extract 12 consecutive timesteps
    sample_raw = df_clean.iloc[20:20 + window_size].copy().reset_index(drop=True)

    # Prepare Scenario 1: Clean Normal Sequence
    seq_normal_raw = sample_raw[feature_cols].values
    seq_normal_scaled = scaler.transform(seq_normal_raw)

    # Prepare Scenario 2: Injected Temperature Spike (+15°C at latest step)
    sample_spike = sample_raw.copy()
    sample_spike.loc[window_size - 1, 'Temperature'] += 15.0
    sample_spike['DewPoint_Spread'] = sample_spike['Temperature'] - sample_spike['DewPoint']
    sample_spike['Temp_Delta'] = sample_spike['Temperature'].diff().fillna(0)
    seq_spike_scaled = scaler.transform(sample_spike[feature_cols].values)

    # Prepare Scenario 3: Physical Law Violation (DewPoint higher than Temperature by 12°C)
    sample_physics = sample_raw.copy()
    sample_physics.loc[window_size - 1, 'DewPoint'] = sample_physics.loc[window_size - 1, 'Temperature'] + 12.0
    sample_physics['DewPoint_Spread'] = sample_physics['Temperature'] - sample_physics['DewPoint']
    seq_physics_scaled = scaler.transform(sample_physics[feature_cols].values)

    test_batch = np.array([seq_normal_scaled, seq_spike_scaled, seq_physics_scaled], dtype=np.float32)

    # 3. Model Inference
    print("\n[3/3] Performing Instant Model Inference & XAI Attribution...")
    reconstructions = lstm_ae.predict(test_batch, verbose=0)
    diff_sq = np.square(test_batch - reconstructions)
    mse_scores = np.mean(diff_sq, axis=(1, 2))

    scenarios = [
        ("SCENARIO 1: Clean Normal Observation", "Real normal atmospheric readings from Jena weather station"),
        ("SCENARIO 2: Sudden Temperature Spike (+15°C)", "Hardware telemetry glitch / sensor impulse spike"),
        ("SCENARIO 3: Physical Law Violation (Tdew > T by 12°C)", "Psychrometric constraint broken: Dew point exceeds ambient temperature")
    ]

    print("\n" + "=" * 70)
    print("                    LIVE INFERENCE TEST RESULTS")
    print("=" * 70)

    for i, (title, desc) in enumerate(scenarios):
        score = float(mse_scores[i])
        is_anom = score > threshold
        status = "[ ANOMALY DETECTED - ALERT! ]" if is_anom else "[ NORMAL - PASS ]"
        
        print(f"\n▶ {title}")
        print(f"  Description:   {desc}")
        print(f"  Anomaly Score: {score:.5f} (Threshold: {threshold:.5f})")
        print(f"  Model Verdict: {status}")

        if is_anom:
            exp = explain_anomaly(i, test_batch, lstm_ae, feature_cols, threshold)
            print("  XAI Root-Cause Attribution:")
            print(f"    * Primary Responsible Sensor: {exp['top_feature']} ({exp['top_feature_percent']}%)")
            sorted_feats = sorted(exp['contributions_percent'].items(), key=lambda x: x[1], reverse=True)[:3]
            for f, c in sorted_feats:
                print(f"      - {f:<18}: {c:>5.1f}% contribution")

    print("\n" + "=" * 70)
    print("  Demonstration completed successfully in < 2 seconds!")
    print("=" * 70)

if __name__ == "__main__":
    run_viva_demo()
