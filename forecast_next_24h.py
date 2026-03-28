import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from core import TimeSeriesModel, prepare_data

if __name__ == "__main__":
    df, data, target = prepare_data()
    train_sz = int(0.8 * len(data))
    
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_X.fit(data[:train_sz])
    scaler_y.fit(target[:train_sz])
    
    X_scaled = scaler_X.transform(data)
    model = TimeSeriesModel(input_size=X_scaled.shape[1])
    model.load_weights("solar_lstm_best.pth")
    model.eval()
    
    seq_len, steps = 24, 24
    last_time = df["time"].iloc[-1]
    ft_times = pd.date_range(start=pd.Timestamp(f"{last_time.date()} 08:00:00"), periods=steps, freq="h")
    
    ft_df = pd.DataFrame({"time": ft_times})
    ft_df["hour"], ft_df["day_of_year"] = ft_df["time"].dt.hour, ft_df["time"].dt.dayofyear
    ft_df["temperature_2m (°C)"], ft_df["Cell_Temp (°C)"], ft_df["pressure_msl"] = df["temperature_2m (°C)"].iloc[-1], df["Cell_Temp (°C)"].iloc[-1], df["pressure_msl"].iloc[-1]
    ft_df["shortwave_radiation (W/m²)"] = ft_df["hour"].apply(lambda h: 800 * np.sin(np.pi * (h - 6) / 13) if 6 <= h <= 18 else 0.0)
    
    preds, curr_seq = [], X_scaled[-seq_len:].copy()
    
    for i in range(steps):
        with torch.no_grad():
            p = model(torch.tensor(curr_seq).unsqueeze(0).float()).numpy()[0][0]
        preds.append(p)
        raw_p = scaler_y.inverse_transform([[p]])[0][0]
        row = [ft_df["temperature_2m (°C)"].iloc[i], ft_df["shortwave_radiation (W/m²)"].iloc[i], ft_df["Cell_Temp (°C)"].iloc[i], ft_df["pressure_msl"].iloc[i], ft_df["hour"].iloc[i], ft_df["day_of_year"].iloc[i], raw_p]
        curr_seq = np.vstack((curr_seq[1:], scaler_X.transform([row])[0]))
    
    inv_preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    total_e, peak_idx = np.sum(inv_preds), np.argmax(inv_preds)
    peak_t, peak_v = ft_times[peak_idx].strftime("%Y-%m-%d %H:%M"), inv_preds[peak_idx]
    
    print(f"\nNext 24 Hour Forecast\n{pd.DataFrame({'time': [t.strftime('%Y-%m-%d %H:%M') for t in ft_times], 'predicted_power_kW': [f'{v:.2f}' for v in inv_preds]}).to_string(index=False)}")
    print(f"\nDaily Energy: {total_e:.2f} kWh\nPeak: {peak_t} ({peak_v:.2f} kW)")
    
    plt.figure(figsize=(12,6))
    plt.plot(ft_times, inv_preds)
    plt.title(f"24h Forecast | Energy: {total_e:.2f} kWh | Peak: {peak_t}")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45), plt.tight_layout(), plt.show()