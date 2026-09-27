import os
import sys
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
    build_comparison_table
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

def main():
    print("=" * 65)
    print(" SIH26073: AWS ANOMALY DETECTION BENCHMARK (BASELINES vs LSTM) ")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. DATA INGESTION & PREPROCESSING
    # -------------------------------------------------------------
    print("\n[Step 1/8] Loading and Preprocessing AWS Data...")
    df_raw = load_data()
    df_selected, feature_cols = preprocess_weather_data(df_raw)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # -------------------------------------------------------------
    # 2. CHRONOLOGICAL SPLIT & ANOMALY INJECTION
    # -------------------------------------------------------------
    print("\n[Step 2/8] Performing Chronological Train-Test Split (80/20)...")
    train_df, test_df = chronological_split(df_selected, train_ratio=0.80)
    print(f"Clean Training Set: {len(train_df):,} rows (100% normal atmospheric data)")
    print(f"Test Set:           {len(test_df):,} rows")

    print("\nInjecting Realistic AWS Sensor Faults into Test Set...")
    test_injected_df, y_test = inject_aws_anomalies(test_df, anomaly_rate=0.05, seed=42)
    normal_count = int(np.sum(y_test == 0))
    anomaly_count = int(np.sum(y_test == 1))
    print(f"Normal Samples:    {normal_count:,} ({normal_count / len(y_test) * 100:.2f}%)")
    print(f"Anomalous Samples: {anomaly_count:,} ({anomaly_count / len(y_test) * 100:.2f}%)")

    # Fit scaler strictly on clean train data to avoid data leakage
    X_train_scaled, X_test_scaled, scaler = fit_and_scale(train_df, test_injected_df, feature_cols)

    # Dictionary to collect results and ROC curve data
    benchmark_results = {}
    roc_plot_data = {}

    # -------------------------------------------------------------
    # 3. BASELINE 1: ISOLATION FOREST
    # -------------------------------------------------------------
    print("\n[Step 3/8] Evaluating Baseline 1: Isolation Forest...")
    iso_forest = build_isolation_forest(contamination=0.05, n_estimators=100, random_state=42)
    iso_forest.fit(X_train_scaled)
    
    if_scores = -iso_forest.score_samples(X_test_scaled)
    if_preds = np.where(iso_forest.predict(X_test_scaled) == -1, 1, 0)
    
    metrics_if = evaluate_predictions(y_test, if_preds, if_scores)
    benchmark_results['Isolation Forest'] = metrics_if
    roc_plot_data['Isolation Forest'] = (y_test, if_scores, metrics_if['roc_auc'])
    print(f"  Isolation Forest -> F1: {metrics_if['f1_score']:.4f}, ROC-AUC: {metrics_if['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 4. BASELINE 2: LOCAL OUTLIER FACTOR (LOF)
    # -------------------------------------------------------------
    print("\n[Step 4/8] Evaluating Baseline 2: Local Outlier Factor (Novelty Mode)...")
    sub_sample_size = 50000
    sub_idx = np.random.choice(len(X_train_scaled), size=sub_sample_size, replace=False)
    
    lof = build_lof(contamination=0.05, n_neighbors=30)
    lof.fit(X_train_scaled[sub_idx])
    
    lof_scores = -lof.score_samples(X_test_scaled)
    lof_preds = np.where(lof.predict(X_test_scaled) == -1, 1, 0)
    
    metrics_lof = evaluate_predictions(y_test, lof_preds, lof_scores)
    benchmark_results['Local Outlier Factor'] = metrics_lof
    roc_plot_data['Local Outlier Factor'] = (y_test, lof_scores, metrics_lof['roc_auc'])
    print(f"  LOF              -> F1: {metrics_lof['f1_score']:.4f}, ROC-AUC: {metrics_lof['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 5. BASELINE 3: DENSE AUTOENCODER (Point-wise)
    # -------------------------------------------------------------
    print("\n[Step 5/8] Training Baseline 3: Dense Autoencoder...")
    dense_ae = build_dense_autoencoder(input_dim=len(feature_cols), learning_rate=1e-3)
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
    dense_ae.fit(
        X_train_scaled, X_train_scaled,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    dense_preds = dense_ae.predict(X_test_scaled, batch_size=512, verbose=0)
    dense_errors = np.mean(np.square(X_test_scaled - dense_preds), axis=1)
    
    # 95th percentile threshold on clean normal validation samples
    dense_val_preds = dense_ae.predict(X_train_scaled[-30000:], batch_size=512, verbose=0)
    dense_val_errors = np.mean(np.square(X_train_scaled[-30000:] - dense_val_preds), axis=1)
    dense_threshold = select_threshold(dense_val_errors, percentile=95.0)
    
    dense_binary_preds = np.where(dense_errors > dense_threshold, 1, 0)
    metrics_dense = evaluate_predictions(y_test, dense_binary_preds, dense_errors)
    benchmark_results['Dense Autoencoder'] = metrics_dense
    roc_plot_data['Dense Autoencoder'] = (y_test, dense_errors, metrics_dense['roc_auc'])
    print(f"  Dense AE         -> F1: {metrics_dense['f1_score']:.4f}, ROC-AUC: {metrics_dense['roc_auc']:.4f} (Threshold: {dense_threshold:.4f})")

    # -------------------------------------------------------------
    # 6. TIME-SERIES WINDOWING & LSTM AUTOENCODER
    # -------------------------------------------------------------
    print("\n[Step 6/8] Building Time-Series Sliding Windows (timesteps=12)...")
    window_size = 12
    # Pre-train split: 80% train, 20% validation sequences
    X_train_seq = create_sequences(X_train_scaled, window_size=window_size)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y=y_test, window_size=window_size, label_mode='any')

    print(f"Training Sequences Shape:   {X_train_seq.shape} (samples, timesteps, features)")
    print(f"Testing Sequences Shape:    {X_test_seq.shape}")
    print(f"Test Sequence Anomalies:    {int(np.sum(y_test_seq)):,} / {len(y_test_seq):,} ({np.mean(y_test_seq)*100:.2f}%)")

    print("\nTraining LSTM Autoencoder on Normal AWS Sequences...")
    lstm_ae = build_lstm_autoencoder(
        timesteps=window_size,
        n_features=len(feature_cols),
        latent_dim=64,
        learning_rate=1e-3
    )
    lstm_ae.summary()

    # Train on normal training sequences with early stopping
    lstm_early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Use 100k representative training sequences for efficient training while preserving patterns
    train_seq_size = min(len(X_train_seq), 120000)
    X_train_sub_seq = X_train_seq[:train_seq_size]

    history = lstm_ae.fit(
        X_train_sub_seq, X_train_sub_seq,
        epochs=12,
        batch_size=512,
        validation_split=0.1,
        callbacks=[lstm_early_stop],
        verbose=1
    )

    # -------------------------------------------------------------
    # 7. RECONSTRUCTION ERROR & EVALUATION
    # -------------------------------------------------------------
    print("\n[Step 7/8] Computing Sequence & Feature-Wise Reconstruction Errors...")
    test_seq_errors, test_feat_errors, X_test_recon = compute_reconstruction_error(lstm_ae, X_test_seq, batch_size=512)
    
    # Validation errors for threshold selection
    val_seq_errors, _, _ = compute_reconstruction_error(lstm_ae, X_train_seq[-15000:], batch_size=512)
    lstm_threshold = select_threshold(val_seq_errors, percentile=95.0)
    print(f"Selected LSTM Anomaly Threshold (95th percentile of normal): {lstm_threshold:.5f}")

    lstm_binary_preds = np.where(test_seq_errors > lstm_threshold, 1, 0)
    metrics_lstm = evaluate_predictions(y_test_seq, lstm_binary_preds, test_seq_errors)
    benchmark_results['LSTM Autoencoder'] = metrics_lstm
    roc_plot_data['LSTM Autoencoder'] = (y_test_seq, test_seq_errors, metrics_lstm['roc_auc'])

    # -------------------------------------------------------------
    # 8. BENCHMARK COMPARISON TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("           FINAL MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 65)
    df_comparison = build_comparison_table(benchmark_results)
    print(df_comparison.to_string(index=False))

    # Save benchmark table as CSV
    table_path = os.path.join(results_dir, "benchmark_metrics.csv")
    df_comparison.to_csv(table_path, index=False)
    print(f"\nBenchmark metrics saved to: {table_path}")

    # -------------------------------------------------------------
    # 9. EXPLAINABILITY (XAI) & DEGRADATION DEMONSTRATIONS
    # -------------------------------------------------------------
    print("\n[XAI Foundation] Evaluating Feature Attribution on Injected Anomalies...")
    # Find sample detected anomalies
    detected_anomaly_indices = np.where((y_test_seq == 1) & (lstm_binary_preds == 1))[0]
    if len(detected_anomaly_indices) > 0:
        sample_idx = detected_anomaly_indices[0]
        explanation = explain_anomaly(sample_idx, X_test_seq, lstm_ae, feature_cols, lstm_threshold)
        print(format_explanation_summary(explanation))
        
        xai_plot_path = os.path.join(results_dir, "xai_explanation_sample.png")
        plot_xai_feature_contributions(explanation, save_path=xai_plot_path)
        print(f"XAI feature attribution plot saved to: {xai_plot_path}")

    print("\n[Sensor Degradation Foundation] Computing Rolling Degradation Indicator...")
    degradation_df = compute_degradation_indicators(test_seq_errors, window_size=144, threshold=lstm_threshold)
    degradation_csv = os.path.join(results_dir, "degradation_indicators.csv")
    degradation_df.head(1000).to_csv(degradation_csv, index=False)
    print(f"Degradation indicator series saved to: {degradation_csv}")

    # -------------------------------------------------------------
    # 10. GENERATING VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n[Visualizations] Generating and Saving Publication Plots...")
    
    # 1. Training history plot
    loss_path = os.path.join(results_dir, "lstm_training_loss.png")
    plot_training_history(history, save_path=loss_path)
    
    # 2. Error distribution plot
    dist_path = os.path.join(results_dir, "error_distribution_threshold.png")
    normal_errs = test_seq_errors[y_test_seq == 0]
    anom_errs = test_seq_errors[y_test_seq == 1]
    plot_error_distribution_and_threshold(normal_errs, anom_errs, lstm_threshold, save_path=dist_path)
    
    # 3. Comparative ROC Curves
    roc_path = os.path.join(results_dir, "model_benchmark_roc.png")
    plot_roc_comparison(roc_plot_data, save_path=roc_path)
    
    # 4. Anomaly timeline
    timeline_path = os.path.join(results_dir, "anomaly_timeline.png")
    plot_anomaly_timeline(y_test_seq, lstm_binary_preds, test_seq_errors, lstm_threshold, sample_range=(0, 500), save_path=timeline_path)

    print(f"\nAll plots saved to: {results_dir}")
    print("=" * 65)
    print(" Pipeline Execution Successfully Completed! ")
    print("=" * 65)

if __name__ == "__main__":
    main()
