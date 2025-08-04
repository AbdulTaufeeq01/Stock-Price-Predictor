import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_utils import download_data, add_technical_indicators, scale_data, create_sequences
from model_utils import stockLSTM, train_model, evaluate_model, predict_future

st.set_page_config(page_title="PyTorch Stock Predictor", layout="wide")
st.title("📉 Stock Price Predictor using PyTorch LSTM")

# Inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL")
n_days = st.slider("Days to Predict", 1, 30, 7)
retrain = st.checkbox("Retrain Model", value=True)

if st.button("Predict"):
    with st.spinner("Fetching data and preparing model..."):
        df = download_data(ticker)
        df = add_technical_indicators(df)
        features = ['Close', 'SMA_20', 'EMA_20', 'RSI']
        scaled, scaler = scale_data(df, features)
        X, y = create_sequences(scaled, lookback=60) # Using a lookback of 60 days

        # Convert to torch tensors
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        split = int(0.8 * len(X))
        X_train, X_test = X_tensor[:split], X_tensor[split:]
        y_train, y_test = y_tensor[:split], y_tensor[split:]

        model = stockLSTM(input_size=X_tensor.shape[2])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if retrain:
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
            train_model(model, train_loader, criterion, optimizer, epochs=10)

        rmse, mae = evaluate_model(model, X_test, y_test)
        st.success(f"✅ Model Performance:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")

        # Forecast
        last_seq = X_tensor[-1]
        future_preds = predict_future(model, last_seq, n_days)
        future_preds = np.array(future_preds).reshape(-1, 1)

        padded = np.zeros((future_preds.shape[0], len(features)))
        padded[:, 0] = future_preds[:, 0]
        inverse_preds = scaler.inverse_transform(padded)[:, 0]

        # Plot
        past = df['Close'].iloc[-60:].tolist()
        all_points = past + inverse_preds.tolist()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(all_points, label="Past + Future")
        ax.axvline(x=len(past) - 1, color='red', linestyle='--', label="Prediction Start")
        ax.set_title(f"{ticker} Closing Price Prediction")
        ax.legend()
        st.pyplot(fig)
