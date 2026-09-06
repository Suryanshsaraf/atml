# SIH26073: AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations

**Subject:** Advanced Topics in Machine Learning (ATML) — Semester 7 B.Tech AI & Data Science  
**Problem Statement ID:** SIH26073 (Smart India Hackathon 2026)  
**Repository:** [Suryanshsaraf/atml](https://github.com/Suryanshsaraf/atml)

---

## 📌 Project Overview

Automatic Weather Stations (AWS) continuously record critical meteorological parameters such as temperature, atmospheric pressure, relative humidity, dew point, and wind speed. Sensor malfunctions, calibration drifts, extreme weather, hardware degradation, and telemetry corruptions frequently produce anomalous observations.

This project focuses on **unsupervised and semi-supervised ML anomaly detection** to automatically flag suspicious weather station observations without relying on heavy domain labels.

---

## 📁 Repository Structure

```
atml/
├── data/
│   ├── jena_climate_2009_2016.csv          # MPI Jena Weather Station Dataset (Primary Benchmark)
│   ├── jena_climate_2009_2016.csv.zip      # Compressed Zip Archive
│   ├── nab_ambient_temperature_system_failure.csv  # Numenta Anomaly Benchmark Sensor Dataset
│   └── chicago_beach_weather_automated_sensors.csv # Chicago Beach AWS Hourly Sensor Dataset
├── download_datasets.py                    # Automated dataset downloader script
├── train_and_evaluate.py                   # Complete ML/DL Training & Evaluation Benchmark Script
├── README.md                               # Project documentation & dataset evaluation guide
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

---

## 📊 Dataset Evaluation & Selection

We evaluated multiple open-source weather sensor datasets across Kaggle, NOAA, NCEI, Zenodo, and municipal open-data portals.

### Selected Primary Dataset: MPI Jena Climate Dataset
* **Source:** Max Planck Institute for Biogeochemistry (MPI-BGC), Jena, Germany
* **Sampling Frequency:** 10-minute intervals (52,560 readings/year)
* **Total Observations:** 420,551 rows (2009–2016)
* **Core Weather Features:**
  * `T (degC)`: Air Temperature
  * `p (mbar)`: Atmospheric Pressure
  * `rh (%)`: Relative Humidity
  * `Tdew (degC)`: Dew Point Temperature
  * `wv (m/s)`: Wind Speed

---

## 🎯 Target Anomaly Injection Modes for Evaluation

1. **Impulse Spikes / Drops:** Instant extreme parameter jumps ($\Delta T > \pm 15^\circ\text{C}$).
2. **Stuck Sensor Readings:** Constant output locked for $L$ consecutive timesteps.
3. **Gradual Calibration Drift:** Linear bias decay over time ($\alpha \cdot t$).
4. **Out-of-Bounds Readings:** Telemetry output exceeding hardware bounds (e.g., $RH > 100\%$).
5. **Physical Correlation Breakdown:** Violation of psychrometric relationships ($T_{\text{dew}} > T$).
6. **Communication / Missing Packet Loss:** Periodic signal dropout and noise injection.

---

## 🤖 Algorithms Benchmarked

* **Isolation Forest (IF)**
* **Local Outlier Factor (LOF)**
* **Deep Autoencoder (Keras Neural Network)**
