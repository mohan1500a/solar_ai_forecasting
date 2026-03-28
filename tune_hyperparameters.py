import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import TensorDataset, DataLoader
from core import TimeSeriesModel, prepare_data, create_sequences

if __name__ == "__main__":
    df, data, target = prepare_data()
    raw_train_sz = int(0.7 * len(data))
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_X.fit(data[:raw_train_sz])
    scaler_y.fit(target[:raw_train_sz])
    X_scaled, y_scaled = scaler_X.transform(data), scaler_y.transform(target)
    
    model_types, num_layers_list = ['LSTM', 'GRU'], [1, 2]
    seq_lengths, hidden_sizes, epochs_list = [12, 24, 48], [32, 64, 128], [30, 50]
    
    results, best_mae, best_config, best_model_state, best_sl = [], float('inf'), {}, None, None
    print("\nGrid Search...")
    search_start = time.time()
    
    idx, total = 1, len(model_types) * len(num_layers_list) * len(seq_lengths) * len(hidden_sizes) * len(epochs_list)
    
    for sl in seq_lengths:
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, sl)
        X_seq, y_seq = torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)
        train_end, val_end = int(0.7 * len(X_seq)), int(0.85 * len(X_seq))
        loader = DataLoader(TensorDataset(X_seq[:train_end], y_seq[:train_end]), batch_size=32, shuffle=True)
        
        for mt in model_types:
            for nl in num_layers_list:
                for hs in hidden_sizes:
                    for ep in epochs_list:
                        print(f"Run {idx}/{total} | Type: {mt}, Layers: {nl}, Seq: {sl}, Hidden: {hs}, Epochs: {ep}")
                        model = TimeSeriesModel(X_seq.shape[2], hs, nl, mt)
                        crit, opt = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001)
                        for _ in range(ep):
                            for bx, by in loader:
                                opt.zero_grad()
                                crit(model(bx), by).backward(), opt.step()
                        
                        model.eval()
                        with torch.no_grad():
                            v_preds = scaler_y.inverse_transform(model(X_seq[train_end:val_end]).numpy())
                            v_actual = scaler_y.inverse_transform(y_seq[train_end:val_end].numpy())
                        
                        mae, rmse, r2 = mean_absolute_error(v_actual, v_preds), np.sqrt(mean_squared_error(v_actual, v_preds)), r2_score(v_actual, v_preds)
                        mask = v_actual > 0.01
                        mape = mean_absolute_percentage_error(v_actual[mask], v_preds[mask]) if mask.sum() > 0 else 0.0
                        
                        print(f"  -> MAE: {mae:.4f} | R²: {r2:.4f}")
                        results.append({"mt": mt, "nl": nl, "sl": sl, "hs": hs, "ep": ep, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2})
                        if mae < best_mae:
                            best_mae, best_config, best_model_state, best_sl = mae, {"mt": mt, "nl": nl, "sl": sl, "hs": hs, "ep": ep}, model.state_dict(), sl
                        idx += 1

    print(f"\nCompleted in {(time.time() - search_start)/60:.2f}m")
    res_df = pd.DataFrame(results).sort_values("mae").head(5)
    print(f"Top 5:\n{res_df.to_string()}")
    torch.save(best_model_state, "solar_lstm_best.pth")
    
    X_fin, y_fin = create_sequences(X_scaled, y_scaled, best_sl)
    X_test = torch.tensor(X_fin[int(0.85 * len(X_fin)):], dtype=torch.float32)
    y_test = torch.tensor(y_fin[int(0.85 * len(X_fin)):], dtype=torch.float32)
    
    fin_model = TimeSeriesModel(X_test.shape[2], best_config["hs"], best_config["nl"], best_config["mt"])
    fin_model.load_state_dict(best_model_state), fin_model.eval()
    with torch.no_grad():
        t_preds = scaler_y.inverse_transform(fin_model(X_test).numpy())
        t_actual = scaler_y.inverse_transform(y_test.numpy())
    
    mask = t_actual > 0.01
    print(f"TEST MAE: {mean_absolute_error(t_actual, t_preds):.5f}\nTEST R²: {r2_score(t_actual, t_preds):.5f}")
