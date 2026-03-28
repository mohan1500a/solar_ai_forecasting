import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import TensorDataset, DataLoader
import time

# =========================
# MODEL DEFINITION
# =========================
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, model_type='LSTM'):
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
# SEQUENCE CREATOR
# =========================
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i : i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    print("Loading dataset...")
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
    
    # Strictly Define 70% bounds for scaler fitting to prevent data leakage
    raw_train_size = int(0.70 * len(data))
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(data[:raw_train_size])
    scaler_y.fit(target[:raw_train_size])
    
    X_scaled = scaler_X.transform(data)
    y_scaled = scaler_y.transform(target)
    
    # HYPERPARAMETER GRID
    model_types = ['LSTM', 'GRU']
    num_layers_list = [1, 2]
    seq_lengths = [12, 24, 48]
    hidden_sizes = [32, 64, 128]
    epochs_list = [30, 50]
    
    results = []
    
    best_val_loss = float('inf')
    best_config = {}
    best_model_state = None
    best_seq_length = None
    
    print("\nStarting Hyperparameter Grid Search...")
    search_start = time.time()
    
    run_idx = 1
    total_runs = len(model_types) * len(num_layers_list) * len(seq_lengths) * len(hidden_sizes) * len(epochs_list)
    
    for seq_len in seq_lengths:
        print(f"\n==========================================")
        print(f"Creating Sequences for Length = {seq_len}")
        print(f"==========================================")
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
        
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_seq = torch.tensor(y_seq, dtype=torch.float32)
        
        # 70% Train, 15% Validation, 15% Test
        n_samples = len(X_seq)
        train_end = int(0.70 * n_samples)
        val_end = int(0.85 * n_samples)
        
        X_train, y_train = X_seq[:train_end], y_seq[:train_end]
        X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]
        
        # We only batch the train_dataset
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        input_size = X_train.shape[2]
        
        for model_type in model_types:
            for nl in num_layers_list:
                for hs in hidden_sizes:
                    for eps in epochs_list:
                        print(f"Run {run_idx}/{total_runs} | Type: {model_type}, Layers: {nl}, Seq: {seq_len}, Hidden: {hs}, Epochs: {eps}")
                        
                        model = TimeSeriesModel(input_size=input_size, hidden_size=hs, num_layers=nl, model_type=model_type)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        
                        # Training Loop
                        model.train()
                        for epoch in range(eps):
                            for batch_X, batch_y in train_loader:
                                optimizer.zero_grad()
                                preds = model(batch_X)
                                loss = criterion(preds, batch_y)
                                loss.backward()
                                optimizer.step()
                        
                        # Validation Loop
                        model.eval()
                        with torch.no_grad():
                            val_preds = model(X_val).numpy()
                            
                        val_preds_inv = scaler_y.inverse_transform(val_preds)
                        val_actual = scaler_y.inverse_transform(y_val.numpy())
                        
                        # Metrics Extraction
                        mae = mean_absolute_error(val_actual, val_preds_inv)
                        rmse = np.sqrt(mean_squared_error(val_actual, val_preds_inv))
                        
                        mask = val_actual > 0.01
                        if mask.sum() > 0:
                            mape = mean_absolute_percentage_error(val_actual[mask], val_preds_inv[mask])
                        else:
                            mape = 0.0
                            
                        r2 = r2_score(val_actual, val_preds_inv)
                        
                        print(f"  -> Validation MAE: {mae:.4f} | MAPE: {mape:.4f} | R²: {r2:.4f}")
                        
                        results.append({
                            "model_type": model_type,
                            "num_layers": nl,
                            "seq_length": seq_len,
                            "hidden_size": hs,
                            "epochs": eps,
                            "val_mae": mae,
                            "val_rmse": rmse,
                            "val_mape": mape,
                            "val_r2": r2
                        })
                        
                        # Track Best Model based on Validation MAE
                        if mae < best_val_loss:
                            best_val_loss = mae
                            best_config = {
                                "model_type": model_type,
                                "num_layers": nl,
                                "seq_length": seq_len, 
                                "hidden_size": hs, 
                                "epochs": eps
                            }
                            best_model_state = model.state_dict()
                            best_seq_length = seq_len
                        
                        run_idx += 1

    print("\n" + "="*50)
    print("GRID SEARCH COMPLETED in {:.2f} minutes".format((time.time() - search_start)/60))
    print("="*50)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_mae", ascending=True).reset_index(drop=True)
    print("\nTop 5 Architectures (by Validation MAE):")
    print(results_df.head(5).to_string())
    
    print("\nSaving best architecture weights -> 'solar_lstm_best.pth'")
    torch.save(best_model_state, "solar_lstm_best.pth")
    
    # =========================
    # FINAL TEST EVALUATION
    # =========================
    print(f"\nEvaluating Best Model {best_config} on the Unseen TEST Holdout (15%)...")
    
    # Rebuild Test sequence for the winning seq_length
    X_seq_best, y_seq_best = create_sequences(X_scaled, y_scaled, best_seq_length)
    n_samples = len(X_seq_best)
    val_end = int(0.85 * n_samples)
    
    X_test_best = torch.tensor(X_seq_best[val_end:], dtype=torch.float32)
    y_test_best = torch.tensor(y_seq_best[val_end:], dtype=torch.float32)
    
    best_model = TimeSeriesModel(
        input_size=X_train.shape[2], 
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"],
        model_type=best_config["model_type"]
    )
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    
    with torch.no_grad():
        test_preds = best_model(X_test_best).numpy()
        
    test_preds_inv = scaler_y.inverse_transform(test_preds)
    test_actual = scaler_y.inverse_transform(y_test_best.numpy())
    
    test_mae = mean_absolute_error(test_actual, test_preds_inv)
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_preds_inv))
    
    mask = test_actual > 0.01
    test_mape = mean_absolute_percentage_error(test_actual[mask], test_preds_inv[mask]) if mask.sum() > 0 else 0.0
    test_r2 = r2_score(test_actual, test_preds_inv)
    
    print(f"TEST MAE : {test_mae:.5f}")
    print(f"TEST RMSE: {test_rmse:.5f}")
    print(f"TEST MAPE: {test_mape:.5f}")
    print(f"TEST R²  : {test_r2:.5f}")
