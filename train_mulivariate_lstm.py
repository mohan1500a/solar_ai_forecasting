import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import TensorDataset, DataLoader

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

    def load_weights(self, path):
        """Robust loader that handles key renaming from 'lstm' to 'rnn'"""
        state_dict = torch.load(path, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            # If the saved model used 'lstm.weight...' but the new class uses 'rnn.weight...'
            new_key = k.replace('lstm.', 'rnn.') if k.startswith('lstm.') else k
            new_state_dict[new_key] = v
        self.load_state_dict(new_state_dict)

# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i : i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    # =========================
    # LOAD DATA
    # =========================
    print("Loading dataset...")
    df = pd.read_csv("solar_data.csv", encoding="latin1").dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"])
    
    # Convert UTC to User's Local Time (+5 hours) so hours match physical sunlight without fractional minutes
    df["time"] = df["time"] + pd.Timedelta(hours=5)
    
    # =========================
    # TIME FEATURES
    # =========================
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    
    # =========================
    # SELECT FEATURES
    # =========================
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
    
    # =========================
    # TRAIN / TEST SPLIT & SCALE (Avoid Data Leakage)
    # =========================
    seq_length = 24
    train_size = int(0.8 * len(data))
    
    # Initialize scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Fit only on training data
    scaler_X.fit(data[:train_size])
    scaler_y.fit(target[:train_size])
    
    # Transform all data
    X_scaled = scaler_X.transform(data)
    y_scaled = scaler_y.transform(target)
    
    # =========================
    # CREATE SEQUENCES
    # =========================
    print("Creating sequences...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)
    
    # Sequence train/test split
    seq_train_size = int(0.8 * len(X_seq))
    
    X_train = X_seq[:seq_train_size]
    y_train = y_seq[:seq_train_size]
    
    X_test = X_seq[seq_train_size:]
    y_test = y_seq[seq_train_size:]
    
    # Implement DataLoader for mini-batching gradient updates
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # =========================
    # INITIALIZE MODEL (Using the best found architecture)
    # =========================
    input_size = X_train.shape[2]
    model = TimeSeriesModel(input_size=input_size, hidden_size=128, num_layers=2, model_type='LSTM')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # =========================
    # TRAINING
    # =========================
    epochs = 50
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")
    
    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "solar_lstm_best.pth")
    print("Model saved: solar_lstm_best.pth")
    
    # =========================
    # EVALUATION
    # =========================
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    predictions = scaler_y.inverse_transform(predictions)
    actual = scaler_y.inverse_transform(y_test.numpy())
    
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    
    # Robust MAPE ignoring night-time zero outputs
    mask = actual > 0.01
    mape = mean_absolute_percentage_error(actual[mask], predictions[mask]) if mask.sum() > 0 else 0.0
    
    print("\nModel Evaluation")
    print(f"MAE  : {mae:.5f}")
    print(f"RMSE : {rmse:.5f}")
    print(f"MAPE : {mape:.5f}")
    print(f"R²   : {r2:.5f}")