# Intelligent Anomaly Detection & XAI for Automatic Weather Stations

**Subject:** Advanced Topics in Machine Learning (ATML — Course Code: 702AI0C018)  
**Program:** B.Tech Artificial Intelligence / AI & Data Science (Semester VII)  
**Repository:** [Suryanshsaraf/atml](https://github.com/Suryanshsaraf/atml)

---

## 📌 Project Overview

Automatic Weather Stations (AWS) continuously record critical meteorological parameters such as temperature, atmospheric pressure, relative humidity, dew point, and wind speed. Sensor malfunctions, calibration drifts, hardware freezes, and telemetry corruptions frequently produce anomalous observations.

This project implements an **unsupervised deep learning pipeline** using an **LSTM Autoencoder** to automatically detect anomalous weather station sensor behaviors and explain which sensor caused the anomaly via **Explainable AI (XAI)** feature attribution.

---

## 📁 Repository Structure

```
atml/
├── data/
│   └── jena_climate_2009_2016.csv          # MPI Jena Weather Station Dataset (Primary Benchmark)
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
│   └── visualization.py                    # Publication-grade plotting
├── results/                                # Generated Benchmark Plots & Metrics
│   ├── benchmark_metrics.csv               # Comparison table of models
│   ├── fault_detection_breakdown.csv       # Per-fault detection rate breakdown
│   ├── model_benchmark_roc.png             # Comparative ROC Curves
│   ├── lstm_training_loss.png              # Training/validation MSE curve
│   ├── error_distribution_threshold.png    # Normal vs Anomaly error distribution
│   ├── anomaly_timeline.png                # Sequence-level detection timeline
│   └── xai_explanation_sample.png          # Sensor feature attribution bar chart
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

## 📊 Final Model Benchmark Comparison (3-Model Benchmark)

All models are trained exclusively on clean normal historical weather data (2009–2014) and evaluated on an identical injected test set (2015–2016, 84,080 sequence windows):

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | Temporal Context |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (Tree-based Baseline)** | 0.8252 | 0.3342 | 0.4757 | 0.6921 | ❌ Static Point-wise (Window Max) |
| **Dense Autoencoder (Deep Learning Static)** | 0.8431 | **0.7603** | **0.7996** | **0.8443** | ❌ Static Point-wise (Window Max) |
| **LSTM Autoencoder (Deep Learning Temporal — Ours)** | **0.9321** | 0.5805 | 0.7154 | 0.8210 | ✅ 12-Step Temporal Trajectory (~2 hrs) |

### Key Takeaway for Viva:
* **LSTM Autoencoder achieves 93.21% Precision**, the highest among all evaluated architectures, ensuring minimal false alarms in operational AWS monitoring.
* Unlike static models that only evaluate instantaneous deviations, the LSTM Autoencoder models temporal continuity over a 2-hour window (12 consecutive 10-minute timesteps), allowing it to capture multi-step temporal dynamics like frozen readings and gradual drifts.

---

## 🎯 Per-Fault Anomaly Detection Breakdown (LSTM Autoencoder)

| Fault Mode | Injected Samples | Detected Samples | Detection Rate (Recall) | Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Impulse Spike** | 635 | 635 | **100.00%** | Extreme temperature jump ($\Delta T = \pm 15^\circ\text{C}$) |
| **Out-of-Bounds** | 841 | 702 | **83.47%** | Telemetry exceeding physical limit ($RH = 145\%$) |
| **Physical Violation** | 841 | 668 | **79.43%** | Psychrometric violation ($T_{\text{dew}} > T + 10^\circ\text{C}$) |
| **Calibration Drift** | 13,620 | 5,034 | **36.96%** | Gradual pressure decay ($-0.2\text{ mbar/step}$) |
| **Stuck Sensor** | 8,033 | 2,692 | **33.51%** | Constant humidity frozen for 2 hours |

---

## 🔍 Explainability (XAI) Foundation

For any detected anomaly, the model computes feature-wise Mean Squared Error attribution:

$$\text{Contribution}_j = \frac{\text{MSE}_j}{\sum_{k=1}^D \text{MSE}_k} \times 100\%$$

```text
--- Explanation for Sequence #92 (ANOMALY DETECTED) ---
Anomaly Score: 0.21544 (Threshold: 0.14080)
Primary Contributor: DewPoint_Spread (53.05%)
Feature Contributions:
  * DewPoint_Spread     : 53.05% (MSE: 1.14291)
  * DewPoint            : 17.84% (MSE: 0.38436)
  * Pressure_Delta      :  8.66% (MSE: 0.18651)
  * Humidity            :  8.00% (MSE: 0.17236)
  * WindSpeed           :  7.33% (MSE: 0.15791)
  * Temp_Delta          :  3.69% (MSE: 0.07956)
```
