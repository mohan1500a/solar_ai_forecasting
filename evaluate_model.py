import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_lstm import SolarLSTM
from sequence import create_sequences
from preprocess import load_and_preprocess

# ==========================
# LOAD DATA & PREPROCESS
# ==========================
print("Loading data...")

# load_and_preprocess handles dropping the 'time' column and scaling all 4 features
scaled_data, scaler = load_and_preprocess("solar_data.csv")

# ==========================
# CREATE SEQUENCES
# ==========================
seq_length = 24

print("Creating sequences...")

# create_sequences from sequence.py already handles taking 4 features in X and 1 feature in y
X, y = create_sequences(scaled_data, seq_length)

X = torch.tensor(X, dtype=torch.float32)

# ==========================
# LOAD MODEL
# ==========================
print("Loading trained model...")

# We must use SolarLSTM here because this is the architecture we saved in train_lstm.py!
model = SolarLSTM()
model.load_state_dict(torch.load("solar_lstm_model.pth"))
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
# We only want to inverse transform the last column (Solar Power)
dummy_pred = np.zeros((len(predictions), scaled_data.shape[1]))
dummy_pred[:, -1] = predictions.flatten()
predictions = scaler.inverse_transform(dummy_pred)[:, -1]

dummy_actual = np.zeros((len(y), scaled_data.shape[1]))
dummy_actual[:, -1] = y.flatten()
actual = scaler.inverse_transform(dummy_actual)[:, -1]

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