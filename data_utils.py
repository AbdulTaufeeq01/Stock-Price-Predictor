import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from indicators import compute_rsi
import numpy as np

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

    df['RSI']=compute_rsi(df['Close'],14) # RSI means Relative Strength Index
    """
    Relative Strength Index is a momemntum oscillator that measures the speed and change of price movements
    If rsi>70 then the stock is overbought and if rsi<30 then the stock is oversold
    It ranges from 0 to 100 and is used to identify overbought or oversold conditions in a stock.
    """

    df.dropna(inplace=True) # Drop rows with NaN values that may have been introduced by rolling calculations
    return df


def scale_data(df,feature_cols):
    """Scales the 'Close' prices in the DataFrame to a range between 0 and 1. 
    This is useful for normalizing the data before training machine learning models.
    This ensures that all features contribute equally to the model's performance.
    Parameters:     
    df (pd.DataFrame): DataFrame containing stock data with a 'Close' column.
    Returns:
    pd.DataFrame: DataFrame with the 'Close' column scaled to a range between 0 and 1.  
    """
    scaler=MinMaxScaler()
    scaled=scaler.fit_transform(df[feature_cols]) # Fit the scaler to the data and transform it

    # scaling is used because it helps the model to perform better and removes the bias of the model.

    return scaler,scaled

def create_sequences(data,lookback=60):
    data=np.array(data) # Convert DataFrame to numpy array
    """
    helps in creating time seres sequeces from the stock data.
    uses a sliding window approach of size lookback
    training the model on the past 60 days of data to predict the next day
    """
    X,y=[], [] # X: list of input sequences (shape: [samples, lookback, features]) y: list of corresponding output values (shape: [samples,])

    for i in range(lookback,len(data)): # Each iteration creates one training sample using the previous lookback values.
        X.append(data[i-lookback:i])
        """
        for eg if lookback=60 and the day is 75 then data[15:75] will be used as 75-60=15 
        this means that the model will use the data from day 15 to day 74 to predict the value of day 75
        """
        y.append(data[i,0])
        """
        y is the value of the next day which is the value of day 75 in this case
        it appends the value of the next day to the Close price column as it has the index of 0
        """
        return np.array(X), np.array(y)




