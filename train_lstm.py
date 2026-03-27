import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from preprocess import load_and_preprocess
from sequence import create_sequences
from sklearn.model_selection import train_test_split
import numpy as np

class SolarLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # input size = 4 features
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :] # Take last hidden state
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    # Load dataset
    print("Loading data...")
    data, scaler = load_and_preprocess("solar_data.csv")
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(data, 24)
    
    # Need to cast to float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SolarLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    # Save model
    print("Saving model to solar_lstm_model.pth")
    torch.save(model.state_dict(), "solar_lstm_model.pth")

    # Evaluate
    model.eval()
    test_mae = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_mae += torch.abs(outputs - batch_y).sum().item()
    test_mae /= len(test_dataset)

    print(f"Test MAE: {test_mae:.4f}")