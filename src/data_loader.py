import os
import pandas as pd

DEFAULT_JENA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

def load_data(data_path=None):
    """
    Loads the Jena Climate dataset from local disk or downloads from Google Cloud mirror.
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "jena_climate_2009_2016.csv")
    
    if os.path.exists(data_path):
        print(f"Loading local dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"Local file not found at {data_path}. Downloading from mirror...")
        df = pd.read_csv(DEFAULT_JENA_URL)
        
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    return df
