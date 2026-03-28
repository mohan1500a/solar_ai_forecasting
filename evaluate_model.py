import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from train_mulivariate_lstm import TimeSeriesModel, create_sequences

from sklearn.preprocessing import MinMaxScaler

# ==========================
# LOAD DATA & PREPROCESS
# ==========================
print("Loading data...")

df = pd.read_csv("solar_data.csv", encoding="latin1").dropna(subset=["time"])
df["time"] = pd.to_datetime(df["time"])
df["time"] = df["time"] + pd.Timedelta(hours=5)
df["hour"] = df["time"].dt.hour
df["day_of_year"] = df["time"].dt.dayofyear

feature_columns = [
    "temperature_2m (°C)",
    "shortwave_radiation (W/m²)",
    "Cell_Temp (°C)",
    "pressure_msl",
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
model = TimeSeriesModel(input_size=input_size, hidden_size=128, num_layers=2, model_type='LSTM')
model.load_weights("solar_lstm_best.pth")
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
r2 = r2_score(actual, predictions)

# Robust MAPE ignoring night-time zero outputs
mask = actual > 0.01
mape = mean_absolute_percentage_error(actual[mask], predictions[mask]) if mask.sum() > 0 else 0.0

print("\nModel Evaluation:")
print(f"MAE  : {mae:.5f}")
print(f"RMSE : {rmse:.5f}")
print(f"MAPE : {mape:.5f}")
print(f"R²   : {r2:.5f}")

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