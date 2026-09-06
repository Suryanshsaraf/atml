import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("=========================================================")
    print(" SIH26073: AWS Anomaly Detection Model Training & Eval ")
    print("=========================================================")

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "jena_climate_2009_2016.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Downloading...")
        jena_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        df_raw = pd.read_csv(jena_url)
    else:
        print(f"Loading local dataset from: {data_path}")
        df_raw = pd.read_csv(data_path)

    # 2. Preprocessing & Feature Engineering
    df = df_raw.copy()
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('Date Time').reset_index(drop=True)

    core_features = {
        'T (degC)': 'Temperature',
        'p (mbar)': 'Pressure',
        'rh (%)': 'Humidity',
        'Tdew (degC)': 'DewPoint',
        'wv (m/s)': 'WindSpeed'
    }
    df_selected = df[['Date Time'] + list(core_features.keys())].rename(columns=core_features)
    df_selected['DewPoint_Spread'] = df_selected['Temperature'] - df_selected['DewPoint']
    df_selected['Temp_Delta'] = df_selected['Temperature'].diff().fillna(0)
    df_selected['Pressure_Delta'] = df_selected['Pressure'].diff().fillna(0)

    hours = df_selected['Date Time'].dt.hour + df_selected['Date Time'].dt.minute / 60.0
    df_selected['Hour_Sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_selected['Hour_Cos'] = np.cos(2 * np.pi * hours / 24.0)

    df_selected = df_selected.dropna().reset_index(drop=True)
    feature_cols = ['Temperature', 'Pressure', 'Humidity', 'DewPoint', 'WindSpeed', 
                    'DewPoint_Spread', 'Temp_Delta', 'Pressure_Delta', 'Hour_Sin', 'Hour_Cos']

    # 3. Train-Test Split
    split_idx = int(len(df_selected) * 0.80)
    train_df = df_selected.iloc[:split_idx].copy()
    test_df = df_selected.iloc[split_idx:].copy()

    # 4. Synthetic Anomaly Injector
    def inject_aws_anomalies(df_test, anomaly_rate=0.05):
        test_data = df_test.copy()
        n_rows = len(test_data)
        n_anomalies = int(n_rows * anomaly_rate)
        labels = np.zeros(n_rows, dtype=int)
        anomaly_indices = np.random.choice(n_rows - 50, size=n_anomalies, replace=False)
        chunk = n_anomalies // 5
        
        # 1. Spike
        for idx in anomaly_indices[0:chunk]:
            test_data.loc[idx, 'Temperature'] += np.random.choice([15.0, -15.0])
            labels[idx] = 1
        # 2. Stuck
        for idx in anomaly_indices[chunk:2*chunk]:
            stuck_val = test_data.loc[idx, 'Humidity']
            for offset in range(12):
                if idx + offset < n_rows:
                    test_data.loc[idx + offset, 'Humidity'] = stuck_val
                    labels[idx + offset] = 1
        # 3. Drift
        for idx in anomaly_indices[2*chunk:3*chunk]:
            for offset in range(18):
                if idx + offset < n_rows:
                    test_data.loc[idx + offset, 'Pressure'] -= 0.2 * (offset + 1)
                    labels[idx + offset] = 1
        # 4. Out-of-bounds
        for idx in anomaly_indices[3*chunk:4*chunk]:
            test_data.loc[idx, 'Humidity'] = 145.0
            labels[idx] = 1
        # 5. Physical Violation
        for idx in anomaly_indices[4*chunk:]:
            test_data.loc[idx, 'DewPoint'] = test_data.loc[idx, 'Temperature'] + 10.0
            labels[idx] = 1

        test_data['DewPoint_Spread'] = test_data['Temperature'] - test_data['DewPoint']
        test_data['Temp_Delta'] = test_data['Temperature'].diff().fillna(0)
        test_data['Pressure_Delta'] = test_data['Pressure'].diff().fillna(0)
        test_data['Label'] = labels
        return test_data

    test_injected_df = inject_aws_anomalies(test_df, anomaly_rate=0.05)
    y_test = test_injected_df['Label'].values

    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_injected_df[feature_cols].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Isolation Forest
    print("\n--- Training Model 1: Isolation Forest ---")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    if_raw_scores = iso_forest.score_samples(X_test)
    if_scores = -if_raw_scores
    if_preds = np.where(iso_forest.predict(X_test) == -1, 1, 0)
    if_auc = roc_auc_score(y_test, if_scores)
    p_if, r_if, f1_if, _ = precision_recall_fscore_support(y_test, if_preds, average='binary')

    # 6. Local Outlier Factor
    print("--- Training Model 2: Local Outlier Factor ---")
    sub_sample_size = 50000
    train_sub_idx = np.random.choice(len(X_train), size=sub_sample_size, replace=False)
    lof = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.05, n_jobs=-1)
    lof.fit(X_train[train_sub_idx])
    lof_scores = -lof.score_samples(X_test)
    lof_preds = np.where(lof.predict(X_test) == -1, 1, 0)
    lof_auc = roc_auc_score(y_test, lof_scores)
    p_lof, r_lof, f1_lof, _ = precision_recall_fscore_support(y_test, lof_preds, average='binary')

    # 7. Deep Autoencoder
    print("--- Training Model 3: Deep Autoencoder ---")
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    enc = Dense(16, activation='relu')(input_layer)
    bottleneck = Dense(4, activation='relu')(enc)
    dec = Dense(16, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='linear')(dec)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X_train, X_train, epochs=25, batch_size=256, validation_split=0.1, callbacks=[early_stop], verbose=1)

    X_test_pred = autoencoder.predict(X_test)
    ae_errors = np.mean(np.square(X_test - X_test_pred), axis=1)
    threshold = np.percentile(ae_errors[y_test == 0], 95)
    ae_preds = np.where(ae_errors > threshold, 1, 0)
    ae_auc = roc_auc_score(y_test, ae_errors)
    p_ae, r_ae, f1_ae, _ = precision_recall_fscore_support(y_test, ae_preds, average='binary')

    # 8. Summary Results
    metrics_df = pd.DataFrame({
        'Model': ['Isolation Forest', 'Local Outlier Factor', 'Deep Autoencoder'],
        'Precision': [p_if, p_lof, p_ae],
        'Recall': [r_if, r_lof, r_ae],
        'F1-Score': [f1_if, f1_lof, f1_ae],
        'ROC-AUC': [if_auc, lof_auc, ae_auc]
    })

    print("\n=========================================================")
    print("          FINAL MODEL BENCHMARK SUMMARY (SIH26073)")
    print("=========================================================")
    print(metrics_df.to_string(index=False))

    # Save plot artifact
    plt.figure(figsize=(10, 5))
    fpr_if, tpr_if, _ = roc_curve(y_test, if_scores)
    fpr_lof, tpr_lof, _ = roc_curve(y_test, lof_scores)
    fpr_ae, tpr_ae, _ = roc_curve(y_test, ae_errors)

    plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {if_auc:.3f})')
    plt.plot(fpr_lof, tpr_lof, label=f'Local Outlier Factor (AUC = {lof_auc:.3f})')
    plt.plot(fpr_ae, tpr_ae, label=f'Deep Autoencoder (AUC = {ae_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves — AWS Anomaly Detection (SIH26073)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(__file__), "model_benchmark_roc.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nROC Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
