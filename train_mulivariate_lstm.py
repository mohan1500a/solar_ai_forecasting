import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import TensorDataset, DataLoader
from core import TimeSeriesModel, prepare_data, create_sequences

if __name__ == "__main__":
    df, data, target = prepare_data()
    seq_len, train_sz = 24, int(0.8 * len(data))
    
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_X.fit(data[:train_sz])
    scaler_y.fit(target[:train_sz])
    
    X_seq, y_seq = create_sequences(scaler_X.transform(data), scaler_y.transform(target), seq_len)
    X_seq, y_seq = torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)
    
    seq_train_sz = int(0.8 * len(X_seq))
    loader = DataLoader(TensorDataset(X_seq[:seq_train_sz], y_seq[:seq_train_sz]), batch_size=32, shuffle=True)
    
    model = TimeSeriesModel(input_size=X_seq.shape[2])
    crit, opt = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        model.train()
        l_sum = 0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            l_sum += loss.item()
        print(f"Epoch {epoch+1}/50 Loss: {l_sum/len(loader):.6f}")
    
    torch.save(model.state_dict(), "solar_lstm_best.pth")
    model.eval()
    with torch.no_grad():
        preds = scaler_y.inverse_transform(model(X_seq[seq_train_sz:]).numpy())
        actual = scaler_y.inverse_transform(y_seq[seq_train_sz:].numpy())
    
    mae, rmse, r2 = mean_absolute_error(actual, preds), np.sqrt(mean_squared_error(actual, preds)), r2_score(actual, preds)
    mask = actual > 0.01
    mape = mean_absolute_percentage_error(actual[mask], preds[mask]) if mask.sum() > 0 else 0.0
    
    print(f"\nModel Evaluation\nMAE: {mae:.5f}\nRMSE: {rmse:.5f}\nMAPE: {mape:.5f}\nR2: {r2:.5f}")