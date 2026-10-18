import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import load_data
from src.preprocessing import preprocess_weather_data, FEATURE_COLS
from src.xai import explain_anomaly

# Page Configuration
st.set_page_config(
    page_title="AWS Anomaly Detection | ATML",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean Academic Theme
st.markdown("""
<style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    model_path = os.path.join(models_dir, "lstm_autoencoder.keras")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata

@st.cache_data
def get_sample_telemetry():
    df_raw = load_data()
    df_clean, _ = preprocess_weather_data(df_raw.iloc[50000:50150])
    return df_clean

# Load artifacts
lstm_model, scaler, metadata = load_model_and_artifacts()

if lstm_model is None:
    st.error("⚠️ Model artifacts not found in `models/` directory! Please run `python train_and_evaluate.py` first.")
    st.stop()

threshold = metadata["threshold"]
window_size = metadata.get("window_size", 12)
feature_cols = metadata.get("feature_cols", FEATURE_COLS)

# Header Section
st.markdown('<p class="main-title">🌤️ Automatic Weather Station (AWS) — Anomaly Detection & XAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">B.Tech 7th Sem ATML Mini-Project | Temporal LSTM Autoencoder Architecture | SIH26073</p>', unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Station Control Panel")
    st.markdown("**Monitoring Station:** Jena AWS-01 (MPI-BGC)")
    st.markdown(f"**Temporal Window:** {window_size} Steps (2 Hours)")
    st.markdown(f"**Anomaly Threshold:** `{threshold:.4f}`")
    st.markdown("*(95th percentile of normal data)*")
    
    st.divider()
    st.subheader("🧪 Select Telemetry Scenario")
    scenario = st.selectbox(
        "Choose Sensor Behavior:",
        [
            "Scenario 1: Clean Normal Observation",
            "Scenario 2: Sudden Temperature Spike (+15°C)",
            "Scenario 3: Psychrometric Violation (Tdew > T)",
            "Scenario 4: Out-of-Bounds Sensor Output (RH = 145%)"
        ]
    )

    st.divider()
    st.markdown("### 📊 Model Architecture")
    st.code("Input(12, 10)\n-> LSTM(64)\n-> Repeat(12)\n-> LSTM(64)\n-> Dense(10)", language="text")

# Extract Base Clean 12-Step Window
df_clean = get_sample_telemetry()
base_sample = df_clean.iloc[20:20 + window_size].copy().reset_index(drop=True)

# Generate Scenario Data
active_sample = base_sample.copy()

if scenario == "Scenario 2: Sudden Temperature Spike (+15°C)":
    active_sample.loc[window_size - 1, 'Temperature'] += 15.0
    active_sample['DewPoint_Spread'] = active_sample['Temperature'] - active_sample['DewPoint']
    active_sample['Temp_Delta'] = active_sample['Temperature'].diff().fillna(0)

elif scenario == "Scenario 3: Psychrometric Violation (Tdew > T)":
    active_sample.loc[window_size - 1, 'DewPoint'] = active_sample.loc[window_size - 1, 'Temperature'] + 12.0
    active_sample['DewPoint_Spread'] = active_sample['Temperature'] - active_sample['DewPoint']

elif scenario == "Scenario 4: Out-of-Bounds Sensor Output (RH = 145%)":
    active_sample.loc[window_size - 1, 'Humidity'] = 145.0

# Display Current Reading KPI Cards (Latest timestep in window)
latest = active_sample.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Temperature", f"{latest['Temperature']:.1f} °C", delta=f"{latest['Temp_Delta']:+.1f} °C" if 'Temp_Delta' in latest else None)
col2.metric("💧 Humidity", f"{latest['Humidity']:.1f} %")
col3.metric("⏲️ Pressure", f"{latest['Pressure']:.1f} mbar")
col4.metric("🌫️ Dew Point Spread", f"{latest['DewPoint_Spread']:.1f} °C")

st.divider()

# Model Inference
seq_raw = active_sample[feature_cols].values
seq_scaled = scaler.transform(seq_raw)
input_tensor = seq_scaled[np.newaxis, ...]

reconstruction = lstm_model.predict(input_tensor, verbose=0)
diff_sq = np.square(input_tensor - reconstruction)
reconstruction_error = float(np.mean(diff_sq))

is_anomaly = reconstruction_error > threshold

# Result Banner
if is_anomaly:
    st.error(f"🚨 **SENSOR ANOMALY DETECTED!** &nbsp;&nbsp;|&nbsp;&nbsp; Reconstruction MSE: **{reconstruction_error:.5f}** &nbsp;>&nbsp; Threshold: **{threshold:.5f}**")
else:
    st.success(f"✅ **ALL SENSORS NORMAL (PASS)** &nbsp;&nbsp;|&nbsp;&nbsp; Reconstruction MSE: **{reconstruction_error:.5f}** &nbsp;≤&nbsp; Threshold: **{threshold:.5f}**")

# Analytics: 2 Columns (XAI Breakdown & Temporal Plot)
c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("🔍 Explainable AI (XAI) Attribution")
    feat_errors = np.mean(diff_sq[0], axis=0)
    total_err = float(np.sum(feat_errors))
    contributions = (feat_errors / (total_err + 1e-12)) * 100.0
    
    top_idx = int(np.argmax(contributions))
    top_sensor = feature_cols[top_idx]
    top_pct = contributions[top_idx]
    
    st.info(f"**Primary Malfunctioning Sensor:** `{top_sensor}` &nbsp;(**{top_pct:.1f}%** contribution to error)")
    
    # Horizontal Bar Chart
    sorted_indices = np.argsort(contributions)
    sorted_feats = [feature_cols[i] for i in sorted_indices]
    sorted_pcts = [contributions[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = ['#EF4444' if f == top_sensor else '#3B82F6' for f in sorted_feats]
    bars = ax.barh(sorted_feats, sorted_pcts, color=colors)
    ax.set_xlabel("Reconstruction Error Contribution (%)", fontsize=9)
    ax.set_xlim(0, max(sorted_pcts) + 15)
    ax.grid(True, alpha=0.2, axis='x')
    
    for bar in bars:
        w = bar.get_width()
        if w > 1.0:
            ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=8, fontweight='bold')
            
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c_right:
    st.subheader("📈 2-Hour Sensor Telemetry Trajectory")
    
    # Plot Trajectory of Primary Sensor
    fig2, ax2 = plt.subplots(figsize=(6, 3.8))
    x_steps = np.arange(1, window_size + 1)
    
    # Determine which parameter to highlight
    if "Temp" in top_sensor:
        param_to_plot = 'Temperature'
        unit = '°C'
    elif "Dew" in top_sensor:
        param_to_plot = 'DewPoint'
        unit = '°C'
    elif "Humidity" in top_sensor:
        param_to_plot = 'Humidity'
        unit = '%'
    else:
        param_to_plot = 'Pressure'
        unit = 'mbar'
        
    ax2.plot(x_steps, active_sample[param_to_plot], marker='o', color='#1D4ED8', lw=2, label=f'{param_to_plot} ({unit})')
    
    if is_anomaly:
        # Highlight anomalous latest step
        ax2.scatter([window_size], [active_sample.loc[window_size - 1, param_to_plot]], color='#DC2626', s=120, zorder=5, label='Fault Point')
        
    ax2.set_xlabel("Observation Step (10-min cadence)", fontsize=9)
    ax2.set_ylabel(f"{param_to_plot} ({unit})", fontsize=9)
    ax2.set_title(f"Sliding Window Trajectory: {param_to_plot}", fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# Bottom Benchmark Expander
with st.expander("📋 View Benchmark Evaluation Table (LSTM vs Baselines)"):
    benchmark_data = {
        'Model Architecture': ['Isolation Forest (Tree-based)', 'Dense Autoencoder (Deep Learning Static)', 'LSTM Autoencoder (Deep Learning Temporal - Ours)'],
        'Precision': [0.8252, 0.8637, 0.9275],
        'Recall': [0.3342, 0.7546, 0.5501],
        'F1-Score': [0.4757, 0.8055, 0.6906],
        'ROC-AUC': [0.6921, 0.8584, 0.8140],
        'Temporal Context': ['❌ Static Point (No Memory)', '❌ Static Point (No Memory)', '✅ 12-Step Window (~2 Hours)']
    }
    st.dataframe(pd.DataFrame(benchmark_data), hide_index=True)
