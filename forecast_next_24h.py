import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =========================
# MODEL
# =========================
# =========================
# MODEL
# =========================
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, model_type='LSTM'):
        super().__init__()
        self.model_type = model_type
        
        # Dropout is only applied if num_layers > 1
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate
            )
        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# =========================
# LOAD DATA
# =========================
print("Loading dataset...")

df = pd.read_csv("solar_data.csv", encoding="latin1").dropna(subset=["time"])

df["time"] = pd.to_datetime(df["time"])
df["time"] = df["time"] + pd.Timedelta(hours=5)

# We keep all data continuous so the LSTM knows what night looks like

# time features
df["hour"] = df["time"].dt.hour
df["day_of_year"] = df["time"].dt.dayofyear

# features
features = [
"temperature_2m (°C)",
"shortwave_radiation (W/m²)",
"Cell_Temp (°C)",
"pressure_msl",
"hour",
"day_of_year",
"Solar_Power (kW)"
]

data = df[features].values

# =========================
# SCALERS
# =========================
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(data)

target = df["Solar_Power (kW)"].values.reshape(-1,1)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(target)

# =========================
# LOAD MODEL
# =========================
input_size = X_scaled.shape[1]
model = TimeSeriesModel(input_size=input_size, hidden_size=128, num_layers=2, model_type='LSTM')
model.load_weights("solar_lstm_best.pth")
model.eval()

print("Model loaded.")

# =========================
# FORECAST
# =========================
seq_length = 24
steps = 24

# Create future time index
last_time = df["time"].iloc[-1]
base_date = last_time.date()
start_time = pd.Timestamp(f"{base_date} 08:00:00")
future_times = pd.date_range(start=start_time, periods=steps, freq="h")

# Build a future raw DataFrame to hold our simulated weather & time
future_df = pd.DataFrame({"time": future_times})
future_df["hour"] = future_df["time"].dt.hour
future_df["day_of_year"] = future_df["time"].dt.dayofyear

# Carry over the last known temperature as a baseline
last_temp = df["temperature_2m (°C)"].iloc[-1]
last_cell = df["Cell_Temp (°C)"].iloc[-1]
last_pressure = df["pressure_msl"].iloc[-1]
future_df["temperature_2m (°C)"] = last_temp
future_df["Cell_Temp (°C)"] = last_cell
future_df["pressure_msl"] = last_pressure

# Simulate a proper Daytime Radiation Curve (Sunrise 06:00, Sunset 18:00)
future_df["shortwave_radiation (W/m²)"] = future_df["hour"].apply(
    lambda h: 800 * np.sin(np.pi * (h - 6) / 13) if 6 <= h <= 18 else 0.0
)
future_df["Solar_Power (kW)"] = 0.0 # Placeholder

future_predictions = []
current_seq = X_scaled[-seq_length:].copy()

for i in range(steps):
    seq_tensor = torch.tensor(current_seq).unsqueeze(0).float()
    
    with torch.no_grad():
        pred = model(seq_tensor).numpy()
        
    power_pred = pred[0][0]
    
    # We let the model predict raw values directly without constraints
    # (Removed zero-clamping at user's request)
        
    future_predictions.append(power_pred)
    
    # Inverse transform the power specifically before doing a master Row Scaler transformation
    raw_power_pred = scaler_y.inverse_transform([[power_pred]])[0][0]
    
    # Grab the true raw future weather parameters for THIS step
    next_raw_row = [
        future_df["temperature_2m (°C)"].iloc[i],
        future_df["shortwave_radiation (W/m²)"].iloc[i],
        future_df["Cell_Temp (°C)"].iloc[i],
        future_df["pressure_msl"].iloc[i],
        future_df["hour"].iloc[i],
        future_df["day_of_year"].iloc[i],
        raw_power_pred  # Plug in our AI's Unscaled Physical Prediction
    ]
    
    # Scale just this single dynamic row to match the sequence dimensions
    next_scaled_row = scaler_X.transform([next_raw_row])[0]
    
    # Shift sequence forward for the next step 
    current_seq = np.vstack((current_seq[1:], next_scaled_row))

# =========================
# INVERSE SCALE
# =========================
future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler_y.inverse_transform(future_predictions)

# =========================
# PRINT RESULTS
# =========================
forecast_df = pd.DataFrame({
    "time": [t.strftime("%Y-%m-%d %H:%M") for t in future_times],
    "predicted_power_kW": [f"{p:.2f}" for p in future_predictions.flatten()]
})

# Calculate Total Energy in kWh (Power * 1 hour interval)
total_energy = future_predictions.flatten().sum()

peak_idx = np.argmax(future_predictions)
peak_time_str = future_times[peak_idx].strftime("%Y-%m-%d %H:%M:%S")
peak_power_val = future_predictions[peak_idx][0]

print("\nNext 24 Hour Forecast\n")
print(forecast_df.to_string(index=False))

print("\nPredicted Daily Energy Generation\n")
print(f"Total Energy (Next 24h): {total_energy:.2f} kWh")
print(f"Peak Time: {peak_time_str} ({peak_power_val:.2f} kW)\n")

# =========================
# PLOT
# =========================
import matplotlib.dates as mdates

plt.figure(figsize=(12,6))

plt.plot(future_times, future_predictions)

start_date = future_times[0].strftime("%b %d, %Y")
end_date = future_times[-1].strftime("%b %d, %Y")

if start_date == end_date:
    date_str = start_date
else:
    date_str = f"{start_date} - {end_date}"

plt.title(f"Next 24 Hour Solar Power Forecast\n(Date: {date_str}) | Total Energy: {total_energy:.2f} kWh\nPeak Time: {peak_time_str}", fontsize=14)
plt.xlabel("Time (Hours)", fontsize=11)
plt.ylabel("Solar Power (kW)", fontsize=11)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()

plt.show()