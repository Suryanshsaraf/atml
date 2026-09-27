import numpy as np
import pandas as pd

def inject_aws_anomalies(df_test, anomaly_rate=0.05, seed=42):
    """
    Simulates 5 realistic Automatic Weather Station failure modes:
    1. Sudden Spikes / Drops (Temperature jump)
    2. Stuck Sensor Readings (Constant Relative Humidity for 12 steps / 2 hours)
    3. Gradual Calibration Drift (Continuous pressure sensor decay over 18 steps / 3 hours)
    4. Out-of-bounds Readings (Humidity = 145%)
    5. Physical Law Violations (DewPoint exceeding Temperature by 10°C)
    
    Generates exact ground-truth labels (0 = Normal, 1 = Anomaly).
    """
    rng = np.random.RandomState(seed)
    test_data = df_test.copy().reset_index(drop=True)
    n_rows = len(test_data)
    n_anomalies = int(n_rows * anomaly_rate)
    
    labels = np.zeros(n_rows, dtype=int)
    # Avoid selecting indices right at the end to accommodate multi-step faults
    anomaly_indices = rng.choice(n_rows - 50, size=n_anomalies, replace=False)
    
    chunk = n_anomalies // 5
    spike_idx = anomaly_indices[0:chunk]
    stuck_idx = anomaly_indices[chunk:2*chunk]
    drift_idx = anomaly_indices[2*chunk:3*chunk]
    bound_idx = anomaly_indices[3*chunk:4*chunk]
    physics_idx = anomaly_indices[4*chunk:]
    
    # 1. Sudden Temperature Spike / Drop (+/- 15°C)
    for idx in spike_idx:
        test_data.loc[idx, 'Temperature'] += rng.choice([15.0, -15.0])
        labels[idx] = 1
        
    # 2. Stuck Sensor (Humidity constant for 12 timesteps / 2 hours)
    for idx in stuck_idx:
        stuck_val = test_data.loc[idx, 'Humidity']
        for offset in range(12):
            if idx + offset < n_rows:
                test_data.loc[idx + offset, 'Humidity'] = stuck_val
                labels[idx + offset] = 1
                
    # 3. Gradual Calibration Drift (-0.2 mbar/step for 18 timesteps / 3 hours)
    for idx in drift_idx:
        for offset in range(18):
            if idx + offset < n_rows:
                test_data.loc[idx + offset, 'Pressure'] -= 0.2 * (offset + 1)
                labels[idx + offset] = 1
                
    # 4. Out-of-bounds Sensor Output (Humidity = 145%)
    for idx in bound_idx:
        test_data.loc[idx, 'Humidity'] = 145.0
        labels[idx] = 1
        
    # 5. Physical Law Violation (DewPoint higher than Temperature)
    for idx in physics_idx:
        test_data.loc[idx, 'DewPoint'] = test_data.loc[idx, 'Temperature'] + 10.0
        labels[idx] = 1
        
    # Recalculate derived features post-injection to reflect real sensor readings
    test_data['DewPoint_Spread'] = test_data['Temperature'] - test_data['DewPoint']
    test_data['Temp_Delta'] = test_data['Temperature'].diff().fillna(0)
    test_data['Pressure_Delta'] = test_data['Pressure'].diff().fillna(0)
    
    test_data['Label'] = labels
    return test_data, labels
