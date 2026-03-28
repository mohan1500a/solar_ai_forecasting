import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, model_type='LSTM'):
        super().__init__()
        dr = 0.2 if num_layers > 1 else 0.0
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dr)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dr)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
    def load_weights(self, path):
        sd = torch.load(path, weights_only=True)
        nsd = { (k.replace('lstm.', 'rnn.') if k.startswith('lstm.') else k): v for k, v in sd.items() }
        self.load_state_dict(nsd)

def prepare_data(path="solar_data.csv"):
    df = pd.read_csv(path, encoding="latin1").dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=5)
    df["hour"], df["day_of_year"] = df["time"].dt.hour, df["time"].dt.dayofyear
    cols = ["temperature_2m (°C)", "shortwave_radiation (W/m²)", "Cell_Temp (°C)", "pressure_msl", "hour", "day_of_year", "Solar_Power (kW)"]
    return df, df[cols].values, df["Solar_Power (kW)"].values.reshape(-1, 1)

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)
