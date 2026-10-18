# SIH26073: AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations

**Subject:** Advanced Topics in Machine Learning (ATML) — Semester 7 B.Tech AI & Data Science  
**Problem Statement ID:** SIH26073 (Smart India Hackathon 2026 / MoES / IMD)  
**Repository:** [Suryanshsaraf/atml](https://github.com/Suryanshsaraf/atml)

---

## 📌 Project Overview

Automatic Weather Stations (AWS) continuously record critical meteorological parameters such as temperature, atmospheric pressure, relative humidity, dew point, and wind speed. Sensor malfunctions, calibration drifts, extreme weather, hardware degradation, and telemetry corruptions frequently produce anomalous observations.

This project implements an **unsupervised machine learning & deep learning pipeline** to automatically detect anomalous weather station sensor behaviors, explain which sensor caused the anomaly (XAI), and monitor sensor degradation trends over time.

---

## 📁 Repository Structure

```
atml/
├── data/
│   ├── jena_climate_2009_2016.csv          # MPI Jena Weather Station Dataset (Primary Benchmark)
│   ├── jena_climate_2009_2016.csv.zip      # Compressed Zip Archive
│   ├── nab_ambient_temperature_system_failure.csv  # Numenta Anomaly Benchmark Sensor Dataset
│   └── chicago_beach_weather_automated_sensors.csv # Chicago Beach AWS Hourly Sensor Dataset
├── models/                                 # Serialized Trained Model Artifacts
│   ├── lstm_autoencoder.keras              # Saved LSTM Autoencoder weights
│   ├── scaler.pkl                          # Fitted StandardScaler on clean training data
│   └── metadata.json                       # Anomaly threshold & feature definitions
├── src/                                    # Modular Source Package
│   ├── __init__.py
│   ├── data_loader.py                      # Data ingestion & caching
│   ├── preprocessing.py                    # Feature engineering & scaling
│   ├── anomaly_injector.py                 # Realistic AWS hardware fault injector
│   ├── sequence_utils.py                   # Time-series sliding window generator
│   ├── models.py                           # Baseline & LSTM Autoencoder models
│   ├── evaluator.py                        # Metric calculation & threshold selection
│   ├── xai.py                              # Feature attribution explainability
│   ├── degradation.py                      # Rolling degradation trend indicator
│   └── visualization.py                    # Publication-grade plotting
├── results/                                # Generated Benchmark Plots & Metrics
│   ├── benchmark_metrics.csv               # Comparison table of all models
│   ├── fault_detection_breakdown.csv       # Per-fault detection rate breakdown
│   ├── model_benchmark_roc.png             # Comparative ROC Curves
│   ├── lstm_training_loss.png              # Training/validation MSE curve
│   ├── error_distribution_threshold.png    # Normal vs Anomaly error distribution
│   ├── anomaly_timeline.png                # Sequence-level detection timeline
│   ├── xai_explanation_sample.png          # Sensor feature attribution bar chart
│   └── degradation_indicators.csv          # Rolling degradation indicators
├── paper/                                  # IEEE Conference Research Paper Source
│   ├── paper.tex                           # Complete ready-to-compile LaTeX paper
│   └── figures/                            # High-resolution embedded figures
├── ATML_MiniProject.ipynb                  # Interactive Google Colab Notebook (Fully Executed)
├── app.py                                  # Interactive Streamlit Monitoring Dashboard
├── demo.py                                 # Instant Viva Demo Script (< 2s execution)
├── download_datasets.py                    # Automated dataset downloader script
├── train_and_evaluate.py                   # End-to-End Pipeline & Evaluation Script
├── requirements.txt                        # Pinned dependencies
├── README.md                               # Project documentation & evaluation report
└── .gitignore                              # Git ignore rules
```

---

## 🖥️ Interactive Streamlit Monitoring Dashboard

Launch the operational AWS anomaly detection and XAI dashboard in one command:

```bash
streamlit run app.py
```

* **Live Sensor KPIs:** Displays real-time Temperature ($^\circ\text{C}$), Pressure ($\text{mbar}$), Humidity ($\%$), and Dew Point Spread ($^\circ\text{C}$).
* **Telemetry Scenarios:** Switch between Normal Observation, Sudden Temperature Spike (+15°C), Psychrometric Violation ($T_{\text{dew}} > T$), and Out-of-Bounds Sensor Output ($RH=145\%$).
* **XAI Diagnostics:** Visualizes feature-wise reconstruction error attribution to pinpoint the responsible sensor probe.
* **Sliding Window Trajectory:** 12-step (2-hour) sequence plot highlighting the anomalous timestamp.

---

## ⚡ Quick Viva Demo CLI (< 2 Seconds)

To run the lightweight command-line demonstration without launching a browser:

```bash
python demo.py
```

This loads the serialized model and scaler, tests 3 real scenarios (Normal Observation, Temperature Impulse Spike, and Psychrometric Law Violation), and outputs the decision and XAI sensor breakdown in ~2 seconds.

---

## 🚀 Running on Google Colab / Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute full training and benchmarking script
python train_and_evaluate.py
```

Or open `ATML_MiniProject.ipynb` directly in Google Colab and run all cells sequentially.

---

## 📊 Final Model Benchmark Comparison (SIH26073 / ATML)

All models are trained exclusively on clean normal historical weather data (2009–2014) and evaluated on an identical injected test set (2015–2016, 84,080 sequence windows):

| Model | Precision | Recall | F1-Score | ROC-AUC | Anomaly Modeling |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 0.8252 | 0.3342 | 0.4757 | 0.6921 | Point-wise Outlier (Window Max) |
| **Local Outlier Factor (LOF)** | 0.8142 | **0.8335** | **0.8237** | **0.8695** | Density-based Novelty (Window Max) |
| **Dense Autoencoder** | 0.8637 | 0.7546 | 0.8055 | 0.8584 | Point-wise Reconstruction (Window Max) |
| **LSTM Autoencoder (Ours)** | **0.9275** | 0.5501 | 0.6906 | 0.8140 | ✅ 12-Step Temporal Trajectory (~2 hrs) |

### Key Takeaway for Viva:
* **LSTM Autoencoder achieves 92.75% Precision**, the highest among all evaluated architectures, ensuring minimal false alarms in mission-critical AWS monitoring.
* Unlike point models that only evaluate instantaneous deviations, the LSTM Autoencoder models temporal continuity over a 2-hour window (12 consecutive 10-minute timesteps), allowing it to reconstruct temporal dynamics and identify sensor hangs and calibration decays.

---

## 🎯 Per-Fault Anomaly Detection Breakdown (LSTM Autoencoder)

| Fault Mode | Injected Samples | Detected Samples | Detection Rate (Recall) | Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Impulse Spike** | 635 | 635 | **100.00%** | Extreme temperature jump ($\Delta T = \pm 15^\circ\text{C}$) |
| **Out-of-Bounds** | 841 | 622 | **73.96%** | Telemetry exceeding physical limit ($RH = 145\%$) |
| **Physical Violation** | 841 | 615 | **73.13%** | Psychrometric violation ($T_{\text{dew}} > T + 10^\circ\text{C}$) |
| **Calibration Drift** | 13,620 | 4,778 | **35.08%** | Gradual pressure decay ($-0.2\text{ mbar/step}$) |
| **Stuck Sensor** | 8,033 | 2,558 | **31.84%** | Constant humidity frozen for 2 hours |

---

## 🔍 Explainability (XAI) Foundation

For any detected anomaly, the model computes feature-wise Mean Squared Error attribution:

$$\text{Contribution}_j = \frac{\text{MSE}_j}{\sum_{k=1}^D \text{MSE}_k} \times 100\%$$

```text
--- Explanation for Sequence #92 (ANOMALY DETECTED) ---
Anomaly Score: 0.21166 (Threshold: 0.15411)
Primary Contributor: DewPoint_Spread (61.25%)
Feature Contributions:
  * DewPoint_Spread     : 61.25% (MSE: 1.29651)
  * DewPoint            : 19.35% (MSE: 0.40963)
  * Temp_Delta          :  4.84% (MSE: 0.10245)
  * Pressure_Delta      :  4.82% (MSE: 0.10199)
  * WindSpeed           :  4.20% (MSE: 0.08887)
```
