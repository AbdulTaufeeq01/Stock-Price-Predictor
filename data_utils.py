import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def download_data(ticker, start_date='2015-01-01',end='2024-12-31'):
    """
    Downloads historical stock data for a given ticker symbol.

    Parameters:
    ticker (str): The stock symbol like AAPL for apple.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data.
    """
    df = yf.download(ticker, start=start_date, end=end)
    df.reset_index(inplace=True) # Reset index to have 'Date' as a column first date was considered as index
    return df
"""
Example usage:
ticker='GOOG'
df=download_data(ticker)
print(df.head(10))

Google opened at $26.01
It closed at $26.22
Around 28 million shares were traded on that day
"""
def add_technical_indicators(df):
    """
    Technical indicators are used to analyze stock price movements and trends.
    its used to predict future price movements as raw prices may not be sufficient to
    predict short-term and long-term price movements.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing stock data with 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
    """
    df['SMA_20']=df['Close'].rolling(window=20).mean() # 20-day Simple Moving Average
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    """
    These are used to identify the average price over a 20-day and 50-day period.
    Used only for the closing price of the stock.
    SMA-20 means simple moving average for 20 days.
    SMA-50 means simple moving average for 50 days.

    This smoothes out short-term fluctuations in the stock price,
    allowing traders to focus on longer-term trends.
    They help to identify long-term trends like bullish or bearish trends.
    """

    df['EMA_20']=df['Close'].ewm(span=20).mean() # 20-day Exponential Moving Average
    df['EMA_50'] = df['Close'].ewm(span=50).mean() # 50-day Exponential Moving Average

    """
    Exponential Moving Averages (EMAs) are similar to SMAs but give more weight to recent prices.
    This makes them more responsive to recent price movements.
    Good in detecting momentums shifts in the stock price.
    EMA-20 means exponential moving average for 20 days.
    EMA-50 means exponential moving average for 50 days.
    """
#rolling gives equal weights to all the values in the window  so its used in sma_20 to give equal weights to identify long term trends
#ewm gives more weight to recent values in the window and the old values fade exponentially so the most recent values have more weight

    df['RSI']=compute_rsi(df['Close'],window=20) # RSI means Relative Strength Index
    """
    Relative Strength Index is a momemntum oscillator that measures the speed and change of price movements
    If rsi>70 then the stock is overbought and if rsi<30 then the stock is oversold
    It ranges from 0 to 100 and is used to identify overbought or oversold conditions in a stock.
    """

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
       
    avg_gain=gain.rolling(window=period).mead() # Average Gain over the period
    avg_loss=loss.rolling(window=period).mean() # Average Loss over the period

    rs=avg_gain/avg_loss # Relative Strength
    rsi=100-(100/(1+rs)) # Relative Strength Index calculation

    return rsi

def scale_data(df)