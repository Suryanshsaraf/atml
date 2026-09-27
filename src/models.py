from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

def build_isolation_forest(contamination=0.05, n_estimators=100, random_state=42):
    """
    Constructs an Isolation Forest baseline model for point-wise anomaly detection.
    """
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

def build_lof(contamination=0.05, n_neighbors=30):
    """
    Constructs a Local Outlier Factor (LOF) model configured in novelty detection mode.
    """
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        contamination=contamination,
        n_jobs=-1
    )

def build_dense_autoencoder(input_dim=10, learning_rate=1e-3):
    """
    Constructs the existing baseline Dense Autoencoder architecture:
    Input (10) -> Dense(16, relu) -> Bottleneck(4, relu) -> Dense(16, relu) -> Output(10, linear)
    """
    input_layer = Input(shape=(input_dim,), name="Dense_AE_Input")
    enc1 = Dense(16, activation='relu', name="Encoder_Dense1")(input_layer)
    bottleneck = Dense(4, activation='relu', name="Bottleneck_Latent")(enc1)
    dec1 = Dense(16, activation='relu', name="Decoder_Dense1")(bottleneck)
    output_layer = Dense(input_dim, activation='linear', name="Reconstructed_Output")(dec1)

    model = Model(inputs=input_layer, outputs=output_layer, name="Dense_Autoencoder")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def build_lstm_autoencoder(timesteps=12, n_features=10, latent_dim=64, learning_rate=1e-3):
    """
    Constructs an LSTM Autoencoder for time-series anomaly detection.
    
    Architecture:
    -------------
    Input: (timesteps, n_features)
    Encoder: LSTM(latent_dim) -> compresses temporal sequence into fixed-length latent vector
    Bridge:  RepeatVector(timesteps) -> distributes latent vector across sequence timesteps
    Decoder: LSTM(latent_dim, return_sequences=True) -> reconstructs temporal hidden states
    Output:  TimeDistributed(Dense(n_features)) -> reconstructs original multi-sensor sequence
    """
    input_layer = Input(shape=(timesteps, n_features), name="LSTM_AE_Input")
    
    # LSTM Encoder
    encoded = LSTM(latent_dim, activation='tanh', recurrent_activation='sigmoid', name="LSTM_Encoder")(input_layer)
    
    # Latent Representation Repeat Vector
    repeated = RepeatVector(timesteps, name="Repeat_Latent")(encoded)
    
    # LSTM Decoder
    decoded = LSTM(latent_dim, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name="LSTM_Decoder")(repeated)
    
    # Reconstructed Output across all timesteps
    output_layer = TimeDistributed(Dense(n_features, activation='linear'), name="TimeDistributed_Dense")(decoded)
    
    model = Model(inputs=input_layer, outputs=output_layer, name="LSTM_Autoencoder")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
