def compute_rsi(series,period=14):
    """
    Computes the Relative Strength Index (RSI) for a given series of stock prices.

    Parameters:
    series (pd.Series): Series of stock prices.
    period (int): The number of periods to use for RSI calculation, default is 1.
    """
    delta=series.diff() # Calculate the difference between consecutive prices usually the close price column
    

    """
    formula for rs= (average gain over n periods) / average loss over n periods
    
    rsi=100-(100/1+rs)
    Average Gain: Mean of all positive price changes over the last 14 days.

    Average Loss: Mean of all negative price changes over the same period.

    If gains dominate → RSI increases.

    If losses dominate → RSI decreases.
    """
    # rsi reflects the market sentiment which is not directly visible in raw price data.

    gain = delta.clip(lower=0)  # Positive changes (gains) replaces all the negative values with zero as they are not considered in positive changes
    loss=-delta.clip(upper=0) # negative changes(losses) replaces all the positive values with zero as they are not considered in negative changes
       
    avg_gain=gain.rolling(window=period).mean() # Average Gain over the period
    avg_loss=loss.rolling(window=period).mean() # Average Loss over the period

    rs=avg_gain/avg_loss # Relative Strength
    rsi=100-(100/(1+rs)) # Relative Strength Index calculation

    return rsi
