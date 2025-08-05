import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    """
    Computes the Relative Strength Index (RSI) for a given series of stock prices.

    Parameters:
    series (pd.Series or array-like): Series of stock prices (e.g., 'Close' prices).
    period (int): The number of periods to use for RSI calculation, default is 14.

    Returns:
    pd.Series: RSI values for the input series.

    Raises:
    ValueError: If the input is empty or period is invalid.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    # Convert array-like inputs to pd.Series
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception as e:
            raise TypeError(f"Input 'series' must be convertible to a pandas Series, got {type(series)}: {str(e)}")

    if series.empty or len(series) < period:
        raise ValueError(f"Input series is empty or too short for period {period}")

    delta = series.diff()  # Calculate price differences

    gain = delta.clip(lower=0)  # Positive changes (gains)
    loss = -delta.clip(upper=0)  # Negative changes (losses)

    avg_gain = gain.rolling(window=period).mean()  # Average gain over period
    avg_loss = loss.rolling(window=period).mean()  # Average loss over period

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # Relative Strength Index

    return rsi