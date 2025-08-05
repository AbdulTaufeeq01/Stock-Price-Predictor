import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class stockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3):
        """
        Initializes an LSTM model for stock price prediction.

        Parameters:
        input_size (int): Number of input features (e.g., number of technical indicators).
        hidden_size (int): Number of units in the LSTM hidden layer.
        num_layers (int): Number of LSTM layers.
        """
        super(stockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, lookback, input_size).

        Returns:
        torch.Tensor: Predicted output of shape (batch_size, 1).
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last timestep's output
        return self.fc(out)

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    """
    Trains the LSTM model using the provided data loader.

    Parameters:
    model (nn.Module): The LSTM model to train.
    train_loader (DataLoader): DataLoader with training sequences and labels.
    criterion: Loss function (e.g., MSELoss).
    optimizer: Optimizer (e.g., Adam).
    epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            out = model(seq)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    print("Training complete.")

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data using RMSE and MAE metrics.

    Parameters:
    model (nn.Module): The trained LSTM model.
    X_test (torch.Tensor): Test input sequences.
    y_test (torch.Tensor): Test target values.

    Returns:
    tuple: (rmse, mae) where rmse is the root mean squared error and mae is the mean absolute error.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().numpy()
        y_true = y_test.squeeze().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def save_model(model, model_path):
    """
    Saves the model's state dictionary to a file.

    Parameters:
    model (nn.Module): The trained model.
    model_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model, model_path):
    """
    Loads a model’s state dictionary from a file.

    Parameters:
    model (nn.Module): The model to load the state into.
    model_path (str): Path to the saved model file.

    Returns:
    nn.Module: The model with loaded weights.
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model

def predict_future(model, last_seq, n_days=7):
    """
    Predicts future stock prices for a specified number of days using the trained model.
    Only updates the 'Close' price in the sequence, keeping other features (e.g., SMA_20, EMA_20, RSI)
    as their last known values to maintain consistency with training data.

    Parameters:
    model (nn.Module): The trained LSTM model.
    last_seq (torch.Tensor): The last sequence of shape (lookback, features).
    n_days (int): Number of days to predict.

    Returns:
    np.ndarray: Array of predicted 'Close' prices with shape (n_days,).
    """
    model.eval()
    preds = []
    seq = last_seq.clone().detach()  # Shape: (lookback, features)

    for _ in range(n_days):
        with torch.no_grad():
            seq_input = seq.unsqueeze(0)  # Add batch dim: (1, lookback, features)
            pred = model(seq_input).item()  # Predict next 'Close' price
        preds.append(pred)
        
        # Create new input row: copy last row, update only 'Close' (index 0)
        new_row = seq[-1].clone()
        new_row[0] = pred  # Update 'Close' with prediction
        seq = torch.cat((seq[1:], new_row.unsqueeze(0)), dim=0)  # Slide window

    return np.array(preds, dtype=np.float32)