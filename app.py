import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

from data_utils import download_data, add_technical_indicators, scale_data, create_sequences
from model_utils import stockLSTM, train_model, evaluate_model, predict_future

# Fix Windows event loop issues in Streamlit
import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set up Streamlit app
st.set_page_config(page_title="📈 PyTorch Stock Predictor", layout="wide")
st.title("📉 Stock Price Predictor using PyTorch LSTM")

# UI inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL")
n_days = st.slider("Days to Predict", 1, 30, 7)
retrain = st.checkbox("Retrain Model", value=True)

if st.button("Predict"):
    with st.spinner("Fetching data and preparing model..."):
        try:
            # Step 1: Download and preprocess data
            try:
                df = download_data(ticker)
            except ValueError as e:
                st.error(f"❌ Data download failed: {str(e)}")
                st.stop()
            st.write(f"Downloaded data shape: {df.shape}, columns: {list(df.columns)}")

            try:
                df = add_technical_indicators(df)
            except (ValueError, TypeError) as e:
                st.error(f"❌ Error computing technical indicators: {str(e)}")
                st.write(f"DataFrame columns: {list(df.columns)}")
                st.stop()
            st.write(f"DataFrame after technical indicators: shape {df.shape}, columns: {list(df.columns)}")

            features = ['Close', 'SMA_20', 'EMA_20', 'RSI']
            scaler, scaled = scale_data(df, features)

            # Step 2: Create sequences
            X, y = create_sequences(scaled, lookback=60)

            if len(X) == 0:
                st.error("❌ Not enough data to create sequences. Please try a different stock or check data range.")
                st.stop()

            # Step 3: Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

            # Step 4: Train/Test split
            split = int(0.8 * len(X))
            X_train, X_test = X_tensor[:split], X_tensor[split:]
            y_train, y_test = y_tensor[:split], y_tensor[split:]

            # Step 5: Model initialization
            input_size = X_tensor.shape[2]  # Number of features
            model = stockLSTM(input_size=input_size, hidden_size=128, num_layers=3)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            # Step 6: Train the model
            if retrain:
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
                train_model(model, train_loader, criterion, optimizer, epochs=300)

            # Step 7: Evaluate the model
            rmse, mae = evaluate_model(model, X_test, y_test)
            st.success(f"✅ Model Performance:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")

            # Step 8: Predict future prices
            last_seq = X_tensor[-1]  # Shape: (lookback, features)
            future_preds = predict_future(model, last_seq, n_days)

            # Ensure future_preds is a 2D NumPy array
            future_preds = np.array(future_preds, dtype=np.float32)
            if future_preds.ndim == 1:
                future_preds = future_preds.reshape(-1, 1)  # Shape: (n_days, 1)

            # Debug: Print shape for verification
            st.write(f"future_preds shape: {future_preds.shape}, type: {type(future_preds)}, dtype: {future_preds.dtype}")

            # Step 9: Inverse scale the predictions
            padded = np.zeros((n_days, len(features)), dtype=np.float32)  # Shape: (n_days, len(features))
            padded[:, 0] = future_preds.flatten()  # Fill 'Close' column

            # Debug: Print padded shape and type
            st.write(f"padded shape: {padded.shape}, type: {type(padded)}, dtype: {padded.dtype}")

            # Check: Ensure padded matches scaler’s expected input
            expected_features = scaler.n_features_in_
            if padded.shape[1] != expected_features:
                st.error(f"❌ Feature mismatch: Scaler expects {expected_features} features, got {padded.shape[1]}")
                st.stop()

            # Inverse transform to get original scale
            try:
                inverse_preds = scaler.inverse_transform(padded)[:, 0]  # Extract 'Close' column
            except Exception as e:
                st.error(f"❌ Error in inverse_transform: {str(e)}")
                st.write(f"padded type: {type(padded)}, dtype: {padded.dtype}, shape: {padded.shape}")
                st.stop()

            # Step 10: Display predicted prices
            st.subheader(f"Predicted Closing Prices for {ticker.upper()} for the next {n_days} days")
            predicted = inverse_preds.flatten().tolist()
            # Create a DataFrame for the predicted prices
            pred_df = pd.DataFrame({
                "Day": [f"Day {i+1}" for i in range(n_days)],
                "Predicted Price (USD)": [round(price, 2) for price in predicted]
            })
            st.table(pred_df)

            # Step 11: Visualize with Plotly
            past = df['Close'].iloc[-60:].values.tolist()
            all_points = past + predicted
            time_points = list(range(len(past) + len(predicted)))

            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_points[:len(past)],
                y=past,
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=time_points[len(past)-1:],
                y=all_points[len(past)-1:],
                mode='lines',
                name='Predicted Prices',
                line=dict(color='orange', dash='dash')
            ))
            fig.add_vline(
                x=len(past)-1,
                line=dict(color='red', dash='dash', width=2),
                annotation_text="Prediction Start",
                annotation_position="top left"
            )
            fig.update_layout(
                title=f"{ticker.upper()} Closing Price Prediction",
                xaxis_title="Time (Days)",
                yaxis_title="Price (USD)",
                showlegend=True,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()