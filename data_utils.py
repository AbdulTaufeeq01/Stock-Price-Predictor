import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from indicators import compute_rsi
import numpy as np

def download_data(ticker, start_date='2015-01-01', end='2024-12-31'):
    """
    Downloads historical stock data for a given ticker symbol.

    Parameters:
    ticker (str): The stock symbol (e.g., 'AAPL' for Apple).
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data with 'Date' as a column.

    Raises:
    ValueError: If no data is retrieved or 'Close' column is missing.
    """
    df = yf.download(ticker, start=start_date, end=end)
    if df.empty:
        raise ValueError(f"No data retrieved for ticker '{ticker}' between {start_date} and {end}")

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    if 'Close' not in df.columns:
        raise ValueError(f"DataFrame missing 'Close' column for ticker '{ticker}', columns: {list(df.columns)}")
    
    df.reset_index(inplace=True)  # Reset index to have 'Date' as a column
    return df

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame to analyze stock price movements and trends.
    These indicators help predict future price movements, as raw prices may not suffice for
    short-term and long-term trend analysis.

    Parameters:
    df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', and 'Volume' columns.

    Returns:
    pd.DataFrame: DataFrame with added technical indicators (SMA_20, SMA_50, EMA_20, EMA_50, RSI).

    Raises:
    ValueError: If 'Close' column is missing or invalid.
    """
    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column missing in DataFrame, columns: {list(df.columns)}")
    if not isinstance(df['Close'], pd.Series):
        raise TypeError(f"'Close' column is not a pandas Series, got {type(df['Close'])}, columns: {list(df.columns)}")

    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20).mean()  # 20-day Exponential Moving Average
    df['EMA_50'] = df['Close'].ewm(span=50).mean()  # 50-day Exponential Moving Average
    df['RSI'] = compute_rsi(df['Close'], 14)  # 14-day Relative Strength Index
    df.dropna(inplace=True)  # Drop rows with NaN values from rolling calculations
    return df

def scale_data(df, feature_cols):
    """
    Scales the specified feature columns in the DataFrame to a range between 0 and 1.
    Normalization ensures features contribute equally to machine learning model performance.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data with specified feature columns.
    feature_cols (list): List of column names to scale (e.g., ['Close', 'SMA_20', 'EMA_20', 'RSI']).

    Returns:
    tuple: (scaler, scaled_data) where scaler is the fitted MinMaxScaler and scaled_data is the scaled NumPy array.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])  # Fit and transform the data
    return scaler, scaled

def create_sequences(data, lookback):
    """
    Creates time series sequences for training a model using a sliding window approach.
    Each sequence uses 'lookback' timesteps to predict the next day's 'Close' price.

    Parameters:
    data (np.ndarray): Scaled data array with shape (samples, features).
    lookback (int): Number of timesteps to use for each sequence.

    Returns:
    tuple: (X, y) where X is the input sequences (shape: [samples, lookback, features]) and
           y is the corresponding output values (shape: [samples,]).
    """
    data = np.array(data)  # Ensure data is a NumPy array
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])  # Sequence of 'lookback' timesteps
        y.append(data[i, 0])  # Next day's 'Close' price (index 0)
    return np.array(X), np.array(y)