import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath):
    """
    Loads data from CSV, drops the time column, and normalizes the features.
    """
    df = pd.read_csv(filepath)
    
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
        
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    return scaled_data, scaler