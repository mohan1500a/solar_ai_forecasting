import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_mulivariate_lstm import LSTMModel, create_sequences
from preprocess import load_and_preprocess

from sklearn.preprocessing import MinMaxScaler

# ==========================
# LOAD DATA & PREPROCESS
# ==========================
print("Loading data...")

df = pd.read_csv("solar_data.csv")
df["time"] = pd.to_datetime(df["time"])
df["time"] = df["time"] + pd.Timedelta(hours=5)
df["hour"] = df["time"].dt.hour
df["day_of_year"] = df["time"].dt.dayofyear

feature_columns = [
    "temperature_2m (°C)",
    "shortwave_radiation (W/m²)",
    "Cell_Temp (°C)",
    "hour",
    "day_of_year",
    "Solar_Power (kW)"
]

data = df[feature_columns].values
target = df["Solar_Power (kW)"].values.reshape(-1, 1)

train_size = int(0.8 * len(data))
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(data[:train_size])
scaler_y.fit(target[:train_size])

X_scaled = scaler_X.transform(data)
y_scaled = scaler_y.transform(target)

# ==========================
# CREATE SEQUENCES
# ==========================
seq_length = 24

print("Creating sequences...")

# create_sequences from sequence.py
X, y = create_sequences(X_scaled, y_scaled, seq_length)

X = torch.tensor(X, dtype=torch.float32)

# ==========================
# LOAD MODEL
# ==========================
print("Loading trained model...")

input_size = X.shape[2]
model = LSTMModel(input_size=input_size)
model.load_state_dict(torch.load("solar_lstm_v3.pth"))
model.eval()

# ==========================
# PREDICTION
# ==========================
print("Making predictions...")

with torch.no_grad():
    predictions = model(X).numpy()

# ==========================
# INVERSE SCALE
# ==========================
predictions = scaler_y.inverse_transform(predictions)
actual = scaler_y.inverse_transform(y.reshape(-1, 1))

# ==========================
# METRICS
# ==========================
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("\nModel Evaluation:")
print("MAE :", mae)
print("RMSE:", rmse)

# ==========================
# PLOT
# ==========================
plt.figure(figsize=(14,6))

plt.plot(actual[:500], label="Actual Power")
plt.plot(predictions[:500], label="Predicted Power")

plt.title("Solar Power Forecasting (LSTM)")
plt.xlabel("Time Step")
plt.ylabel("Power Output")

plt.legend()
plt.savefig("evaluation_plot.png")
print("Saved evaluation plot to evaluation_plot.png")
plt.show()