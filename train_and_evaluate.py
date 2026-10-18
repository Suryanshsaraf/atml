import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import load_data
from src.preprocessing import preprocess_weather_data, chronological_split, fit_and_scale, FEATURE_COLS
from src.anomaly_injector import inject_aws_anomalies
from src.sequence_utils import create_sequences
from src.models import (
    build_isolation_forest,
    build_lof,
    build_dense_autoencoder,
    build_lstm_autoencoder
)
from src.evaluator import (
    compute_reconstruction_error,
    select_threshold,
    evaluate_predictions,
    build_comparison_table,
    evaluate_by_fault_type
)
from src.xai import explain_anomaly, format_explanation_summary
from src.degradation import compute_degradation_indicators
from src.visualization import (
    plot_training_history,
    plot_error_distribution_and_threshold,
    plot_roc_comparison,
    plot_anomaly_timeline,
    plot_xai_feature_contributions
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def window_max_pooling(point_scores, window_size=12):
    """
    Aggregates point-wise anomaly scores into sequence window scores
    using maximum pooling across each 12-step sliding window.
    """
    return np.max(np.lib.stride_tricks.sliding_window_view(point_scores, window_size), axis=1)

def main():
    print("=" * 68)
    print("  SIH26073 / ATML: AWS ANOMALY DETECTION BENCHMARK PIPELINE  ")
    print("=" * 68)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. DATA INGESTION & PREPROCESSING
    # -------------------------------------------------------------
    print("\n[Step 1/8] Loading and Preprocessing AWS Meteorological Data...")
    df_raw = load_data()
    df_selected, feature_cols = preprocess_weather_data(df_raw)
    print(f"Engineered Features ({len(feature_cols)}): {feature_cols}")

    # -------------------------------------------------------------
    # 2. CHRONOLOGICAL SPLIT & ANOMALY INJECTION
    # -------------------------------------------------------------
    print("\n[Step 2/8] Performing Chronological Train-Test Split (80/20)...")
    train_df, test_df = chronological_split(df_selected, train_ratio=0.80)
    print(f"Clean Training Set (2009–2014): {len(train_df):,} rows (100% normal data)")
    print(f"Test Set (2015–2016):           {len(test_df):,} rows")

    print("\nInjecting 5 Realistic AWS Hardware Sensor Faults into Test Set...")
    test_injected_df, y_test = inject_aws_anomalies(test_df, anomaly_rate=0.05, seed=42)
    normal_count = int(np.sum(y_test == 0))
    anomaly_count = int(np.sum(y_test == 1))
    print(f"Point-wise Normal Samples:    {normal_count:,} ({normal_count / len(y_test) * 100:.2f}%)")
    print(f"Point-wise Anomalous Samples: {anomaly_count:,} ({anomaly_count / len(y_test) * 100:.2f}%)")

    # Fit scaler strictly on clean train data to prevent data leakage
    X_train_scaled, X_test_scaled, scaler = fit_and_scale(train_df, test_injected_df, feature_cols)

    # -------------------------------------------------------------
    # 3. TIME-SERIES SLIDING WINDOWS (2 HOURS = 12 STEPS)
    # -------------------------------------------------------------
    window_size = 12
    print(f"\n[Step 3/8] Building Time-Series Sliding Windows (window_size={window_size})...")
    X_train_seq = create_sequences(X_train_scaled, window_size=window_size)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y=y_test, window_size=window_size, label_mode='any')

    # Aligned fault types for per-fault detection rate
    fault_types_aligned = test_injected_df['Fault_Type'].values[window_size - 1:]

    print(f"Training Sequences: {X_train_seq.shape}")
    print(f"Testing Sequences:  {X_test_seq.shape}")
    print(f"Test Window Anomalies: {int(np.sum(y_test_seq)):,} / {len(y_test_seq):,} ({np.mean(y_test_seq)*100:.2f}%)")

    benchmark_results = {}
    roc_plot_data = {}

    # -------------------------------------------------------------
    # 4. BASELINE 1: ISOLATION FOREST
    # -------------------------------------------------------------
    print("\n[Step 4/8] Evaluating Baseline 1: Isolation Forest...")
    iso_forest = build_isolation_forest(contamination=0.05, n_estimators=100, random_state=42)
    iso_forest.fit(X_train_scaled)
    
    if_point_scores = -iso_forest.score_samples(X_test_scaled)
    if_seq_scores = window_max_pooling(if_point_scores, window_size)
    
    # Validation threshold (95th percentile of normal)
    if_val_scores = -iso_forest.score_samples(X_train_scaled[-15000:])
    if_thresh = np.percentile(if_val_scores, 95.0)
    if_seq_preds = (if_seq_scores > if_thresh).astype(int)
    
    metrics_if = evaluate_predictions(y_test_seq, if_seq_preds, if_seq_scores)
    benchmark_results['Isolation Forest'] = metrics_if
    roc_plot_data['Isolation Forest'] = (y_test_seq, if_seq_scores, metrics_if['roc_auc'])
    print(f"  Isolation Forest -> F1: {metrics_if['f1_score']:.4f}, Precision: {metrics_if['precision']:.4f}, Recall: {metrics_if['recall']:.4f}, ROC-AUC: {metrics_if['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 5. BASELINE 2: LOCAL OUTLIER FACTOR (LOF)
    # -------------------------------------------------------------
    print("\n[Step 5/8] Evaluating Baseline 2: Local Outlier Factor (Novelty Mode)...")
    sub_sample_size = 50000
    sub_idx = np.random.choice(len(X_train_scaled), size=sub_sample_size, replace=False)
    lof = build_lof(contamination=0.05, n_neighbors=30)
    lof.fit(X_train_scaled[sub_idx])
    
    lof_point_scores = -lof.score_samples(X_test_scaled)
    lof_seq_scores = window_max_pooling(lof_point_scores, window_size)
    
    lof_val_scores = -lof.score_samples(X_train_scaled[-15000:])
    lof_thresh = np.percentile(lof_val_scores, 95.0)
    lof_seq_preds = (lof_seq_scores > lof_thresh).astype(int)
    
    metrics_lof = evaluate_predictions(y_test_seq, lof_seq_preds, lof_seq_scores)
    benchmark_results['Local Outlier Factor'] = metrics_lof
    roc_plot_data['Local Outlier Factor'] = (y_test_seq, lof_seq_scores, metrics_lof['roc_auc'])
    print(f"  LOF              -> F1: {metrics_lof['f1_score']:.4f}, Precision: {metrics_lof['precision']:.4f}, Recall: {metrics_lof['recall']:.4f}, ROC-AUC: {metrics_lof['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 6. BASELINE 3: DENSE AUTOENCODER (Static Point-wise AE)
    # -------------------------------------------------------------
    print("\n[Step 6/8] Training Baseline 3: Dense Autoencoder...")
    dense_ae = build_dense_autoencoder(input_dim=len(feature_cols), learning_rate=1e-3)
    early_stop_dense = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    dense_ae.fit(
        X_train_scaled, X_train_scaled,
        epochs=12,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop_dense],
        verbose=1
    )
    dense_preds = dense_ae.predict(X_test_scaled, batch_size=512, verbose=0)
    dense_point_errors = np.mean(np.square(X_test_scaled - dense_preds), axis=1)
    dense_seq_scores = window_max_pooling(dense_point_errors, window_size)
    
    dense_val_preds = dense_ae.predict(X_train_scaled[-20000:], batch_size=512, verbose=0)
    dense_val_errors = np.mean(np.square(X_train_scaled[-20000:] - dense_val_preds), axis=1)
    dense_threshold = select_threshold(dense_val_errors, percentile=95.0)
    dense_seq_preds = (dense_seq_scores > dense_threshold).astype(int)
    
    metrics_dense = evaluate_predictions(y_test_seq, dense_seq_preds, dense_seq_scores)
    benchmark_results['Dense Autoencoder'] = metrics_dense
    roc_plot_data['Dense Autoencoder'] = (y_test_seq, dense_seq_scores, metrics_dense['roc_auc'])
    print(f"  Dense AE         -> F1: {metrics_dense['f1_score']:.4f}, Precision: {metrics_dense['precision']:.4f}, Recall: {metrics_dense['recall']:.4f}, ROC-AUC: {metrics_dense['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 7. PROPOSED MODEL: LSTM AUTOENCODER (Temporal Sequence AE)
    # -------------------------------------------------------------
    print("\n[Step 7/8] Training Proposed LSTM Autoencoder (Temporal Modeling)...")
    lstm_ae = build_lstm_autoencoder(
        timesteps=window_size,
        n_features=len(feature_cols),
        latent_dim=64,
        learning_rate=1e-3
    )
    lstm_ae.summary()

    lstm_early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    train_seq_size = min(len(X_train_seq), 100000)
    X_train_sub = X_train_seq[:train_seq_size]

    history = lstm_ae.fit(
        X_train_sub, X_train_sub,
        epochs=10,
        batch_size=512,
        validation_split=0.1,
        callbacks=[lstm_early_stop],
        verbose=1
    )

    # Compute LSTM reconstruction error
    test_seq_errors, test_feat_errors, X_test_recon = compute_reconstruction_error(lstm_ae, X_test_seq, batch_size=512)
    val_seq_errors, _, _ = compute_reconstruction_error(lstm_ae, X_train_seq[-15000:], batch_size=512)
    lstm_threshold = select_threshold(val_seq_errors, percentile=95.0)
    print(f"Selected LSTM Anomaly Threshold (95th percentile of normal): {lstm_threshold:.5f}")

    lstm_binary_preds = (test_seq_errors > lstm_threshold).astype(int)
    metrics_lstm = evaluate_predictions(y_test_seq, lstm_binary_preds, test_seq_errors)
    benchmark_results['LSTM Autoencoder (Ours)'] = metrics_lstm
    roc_plot_data['LSTM Autoencoder (Ours)'] = (y_test_seq, test_seq_errors, metrics_lstm['roc_auc'])
    print(f"  LSTM AE (Ours)   -> F1: {metrics_lstm['f1_score']:.4f}, Precision: {metrics_lstm['precision']:.4f}, Recall: {metrics_lstm['recall']:.4f}, ROC-AUC: {metrics_lstm['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 8. BENCHMARK COMPARISON TABLE & ARTIFACT PERSISTENCE
    # -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("      FINAL COMPARATIVE BENCHMARK TABLE (IDENTICAL TEST SAMPLES)")
    print("=" * 68)
    df_comparison = build_comparison_table(benchmark_results)
    print(df_comparison.to_string(index=False))

    table_path = os.path.join(results_dir, "benchmark_metrics.csv")
    df_comparison.to_csv(table_path, index=False)
    print(f"\nBenchmark metrics saved to: {table_path}")

    # Per-Fault Anomaly Detection Breakdown
    print("\n" + "=" * 68)
    print("      PER-FAULT ANOMALY DETECTION BREAKDOWN (LSTM AUTOENCODER)")
    print("=" * 68)
    fault_breakdown_df = evaluate_by_fault_type(fault_types_aligned, lstm_binary_preds)
    print(fault_breakdown_df.to_string(index=False))
    fault_table_path = os.path.join(results_dir, "fault_detection_breakdown.csv")
    fault_breakdown_df.to_csv(fault_table_path, index=False)

    # Save Models for Viva Demo
    print("\n[Step 8/8] Saving Model & Scaler Artifacts for Instant Viva Demo...")
    model_save_path = os.path.join(models_dir, "lstm_autoencoder.keras")
    lstm_ae.save(model_save_path)
    scaler_save_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_save_path)

    metadata = {
        "threshold": float(lstm_threshold),
        "window_size": int(window_size),
        "feature_cols": feature_cols,
        "metrics": {
            "precision": float(metrics_lstm["precision"]),
            "recall": float(metrics_lstm["recall"]),
            "f1_score": float(metrics_lstm["f1_score"]),
            "roc_auc": float(metrics_lstm["roc_auc"])
        }
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Artifacts successfully saved to 'models/' directory!")

    # -------------------------------------------------------------
    # 9. XAI ATTRIBUTION & DEGRADATION TRACKING DEMONSTRATIONS
    # -------------------------------------------------------------
    detected_anomaly_indices = np.where((y_test_seq == 1) & (lstm_binary_preds == 1))[0]
    if len(detected_anomaly_indices) > 0:
        sample_idx = detected_anomaly_indices[0]
        explanation = explain_anomaly(sample_idx, X_test_seq, lstm_ae, feature_cols, lstm_threshold)
        print("\n" + format_explanation_summary(explanation))
        
        xai_plot_path = os.path.join(results_dir, "xai_explanation_sample.png")
        plot_xai_feature_contributions(explanation, save_path=xai_plot_path)

    degradation_df = compute_degradation_indicators(test_seq_errors, window_size=144, threshold=lstm_threshold)
    degradation_df.head(1000).to_csv(os.path.join(results_dir, "degradation_indicators.csv"), index=False)

    # Generate Publication Plots
    plot_training_history(history, save_path=os.path.join(results_dir, "lstm_training_loss.png"))
    
    normal_errs = test_seq_errors[y_test_seq == 0]
    anom_errs = test_seq_errors[y_test_seq == 1]
    plot_error_distribution_and_threshold(normal_errs, anom_errs, lstm_threshold, save_path=os.path.join(results_dir, "error_distribution_threshold.png"))
    
    plot_roc_comparison(roc_plot_data, save_path=os.path.join(results_dir, "model_benchmark_roc.png"))
    plot_anomaly_timeline(y_test_seq, lstm_binary_preds, test_seq_errors, lstm_threshold, sample_range=(0, 500), save_path=os.path.join(results_dir, "anomaly_timeline.png"))

    print(f"\nAll publication plots successfully updated in: {results_dir}")
    print("=" * 68)
    print(" Training, Evaluation, and Serialization Complete! ")
    print("=" * 68)

if __name__ == "__main__":
    main()
