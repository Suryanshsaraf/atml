# SIH26073: AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations

**Subject:** Advanced Topics in Machine Learning (ATML) — Semester 7 B.Tech AI & Data Science  
**Problem Statement ID:** SIH26073 (Smart India Hackathon 2026)  
**Repository:** [Suryanshsaraf/atml](https://github.com/Suryanshsaraf/atml)

---

## 📌 Project Overview

Automatic Weather Stations (AWS) continuously record critical meteorological parameters such as temperature, atmospheric pressure, relative humidity, dew point, and wind speed. Sensor malfunctions, calibration drifts, extreme weather, hardware degradation, and telemetry corruptions frequently produce anomalous observations.

This project implements an **unsupervised and semi-supervised machine learning & deep learning pipeline** to automatically detect anomalous weather station sensor behaviors, explain which sensor caused the anomaly (XAI), and monitor sensor degradation trends over time.

---

## 📁 Repository Structure

```
atml/
├── data/
│   ├── jena_climate_2009_2016.csv          # MPI Jena Weather Station Dataset (Primary Benchmark)
│   ├── jena_climate_2009_2016.csv.zip      # Compressed Zip Archive
│   ├── nab_ambient_temperature_system_failure.csv  # Numenta Anomaly Benchmark Sensor Dataset
│   └── chicago_beach_weather_automated_sensors.csv # Chicago Beach AWS Hourly Sensor Dataset
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
│   ├── model_benchmark_roc.png             # Comparative ROC Curves
│   ├── lstm_training_loss.png              # Training/validation MSE curve
│   ├── error_distribution_threshold.png    # Normal vs Anomaly error distribution
│   ├── anomaly_timeline.png                # Sequence-level detection timeline
│   ├── xai_explanation_sample.png          # Sensor feature attribution bar chart
│   └── degradation_indicators.csv          # Rolling degradation indicators
├── ATML_MiniProject.ipynb                  # Interactive Google Colab Notebook
├── download_datasets.py                    # Automated dataset downloader script
├── train_and_evaluate.py                   # End-to-End Pipeline & Evaluation Script
├── README.md                               # Project documentation & evaluation report
└── .gitignore                              # Git ignore rules
```

---

## 🚀 Running on Google Colab

To run the complete training pipeline directly in Google Colab:

```python
# 1. Clone this repository into Colab
!git clone https://github.com/Suryanshsaraf/atml.git
%cd atml

# 2. Execute full training and benchmarking script
!python train_and_evaluate.py
```

Or open `ATML_MiniProject.ipynb` directly in Google Colab and run all cells sequentially.

---

## 📊 Final Model Benchmark Comparison (SIH26073)

All models are trained exclusively on clean normal historical weather data (2009–2014) and evaluated on an injected test split (2015–2016):

| Model | Precision | Recall | F1-Score | ROC-AUC | Temporal Modeling |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 0.5368 | 0.1415 | 0.2239 | 0.6423 | ❌ Point-wise (Static) |
| **Local Outlier Factor (LOF)** | 0.6557 | 0.4198 | 0.5119 | 0.7443 | ❌ Point-wise (Static) |
| **Dense Autoencoder** | 0.6055 | 0.2407 | 0.3445 | 0.7109 | ❌ Point-wise (Static) |
| **LSTM Autoencoder (Ours)** | **0.9342** | **0.5961** | **0.7278** | **0.8213** | ✅ 12-Step Temporal Window (~2 hrs) |

### Key Takeaway:
The **LSTM Autoencoder** substantially outperforms all static baseline models. By modeling the 12-step temporal context (2 hours of history), the LSTM Autoencoder doubles the F1-score (0.7278 vs 0.3445) and achieves **93.42% precision**, successfully distinguishing complex multi-step temporal anomalies (stuck sensors, calibration drift) that point-wise models miss.

---

## 🔍 Explainability (XAI) Foundation

For any detected anomaly, the model computes feature-wise reconstruction error attribution:

```text
--- Explanation for Sequence #92 (ANOMALY DETECTED) ---
Anomaly Score: 0.20492 (Threshold: 0.12848)
Primary Contributor: DewPoint_Spread (58.34%)
Feature Contributions:
  * DewPoint_Spread     : 58.34% (MSE: 1.19558)
  * DewPoint            : 18.19% (MSE: 0.37267)
  * Temp_Delta          : 10.12% (MSE: 0.20730)
  * Humidity            :  4.37% (MSE: 0.08961)
  * Pressure_Delta      :  3.69% (MSE: 0.07561)
```

---

## 🎯 Target Anomaly Injection Modes for Evaluation

1. **Impulse Spikes / Drops:** Instant extreme parameter jumps ($\Delta T > \pm 15^\circ\text{C}$).
2. **Stuck Sensor Readings:** Constant output locked for 12 consecutive timesteps (2 hours).
3. **Gradual Calibration Drift:** Linear decay over 18 steps ($-0.2\text{ mbar/step}$).
4. **Out-of-Bounds Readings:** Telemetry output exceeding physical limits ($RH = 145\%$).
5. **Physical Correlation Breakdown:** Violation of psychrometric laws ($T_{\text{dew}} > T$).
