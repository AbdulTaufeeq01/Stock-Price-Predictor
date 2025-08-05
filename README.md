# Stock Price Predictor
# Stock Price Predictor
Stock Price Predictor

A web-based application for predicting future stock prices using a Long Short-Term Memory (LSTM) neural network implemented in PyTorch. The app fetches historical stock data, computes technical indicators (Simple Moving Average, Exponential Moving Average, and Relative Strength Index), trains an LSTM model, and visualizes predictions using an interactive Plotly graph. Built with Streamlit for an intuitive user interface.
Table of Contents

Features
Installation
Usage
File Structure
Dependencies
How It Works
Troubleshooting
Contributing
License

Features

Data Retrieval: Fetches historical stock data from Yahoo Finance using yfinance.
Technical Indicators: Calculates 20-day and 50-day SMA/EMA, and 14-day RSI to enhance model predictions.
LSTM Model: Uses a PyTorch LSTM to predict future stock prices based on historical data and indicators.
Interactive Visualization: Displays historical and predicted prices in an interactive Plotly graph.
User Interface: Streamlit app allows users to input a stock ticker, select prediction days, and choose whether to retrain the model.
Error Handling: Robust checks for data issues, such as empty datasets or invalid tickers.

Installation

Clone the Repository:
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

See Dependencies for the full list of required packages.

Run the Application:
streamlit run app.py



Usage

Open the Streamlit app in your browser (typically at http://localhost:8501).
Enter a stock ticker (e.g., AAPL for Apple).
Select the number of days to predict (1 to 30, default is 7).
Check "Retrain Model" to train a new LSTM model or uncheck to use the existing model (if saved).
Click "Predict" to fetch data, train the model, and view results.
View the output:
A table showing the predicted closing prices for the next n_days.
An interactive Plotly graph displaying the past 60 days of closing prices and the predicted prices, with a vertical line marking the prediction start.



Example output for AAPL (7-day prediction):



Day
Predicted Price (USD)



Day 1
219.16


Day 2
218.90


Day 3
218.40


Day 4
217.78


Day 5
217.12


Day 6
216.47


Day 7
215.86


File Structure

app.py: Main Streamlit application script. Handles user input, data processing, model training, prediction, and visualization using Plotly.
data_utils.py: Functions for downloading stock data, computing technical indicators, scaling data, and creating time series sequences.
indicators.py: Function to compute the Relative Strength Index (RSI) for a given price series.
model_utils.py: Defines the LSTM model, training, evaluation, and prediction functions using PyTorch.
requirements.txt: List of Python dependencies for the project.
README.md: This file, providing project documentation.

Dependencies

Python 3.8 or higher
streamlit>=1.20.0: For the web interface
yfinance>=0.2.0: For fetching stock data
pandas>=1.5.0: For data manipulation
numpy>=1.23.0: For numerical operations
torch>=2.0.0: For the LSTM model
scikit-learn>=1.2.0: For data scaling
plotly>=5.10.0: For interactive visualizations

Install all dependencies with:
pip install streamlit yfinance pandas numpy torch scikit-learn plotly

How It Works

Data Download: Fetches historical stock data for the specified ticker from Yahoo Finance (data_utils.download_data).
Technical Indicators: Computes SMA (20, 50 days), EMA (20, 50 days), and RSI (14 days) to capture price trends (data_utils.add_technical_indicators).
Data Preprocessing: Scales features (Close, SMA_20, EMA_20, RSI) to [0, 1] and creates time series sequences with a 60-day lookback (data_utils.scale_data, data_utils.create_sequences).
Model Training: Trains a 3-layer LSTM model with 100 hidden units using PyTorch (model_utils.train_model).
Prediction: Predicts the next n_days closing prices using the trained model (model_utils.predict_future).
Visualization: Displays historical and predicted prices in a Plotly graph and a table of predicted prices (app.py).

Troubleshooting

MultiIndex Error: If you see 'Close' column is not a pandas Series or similar, ensure yfinance returns a single-level column index. The download_data function flattens MultiIndex columns to prevent this.
Empty Data: If no data is retrieved, check the ticker (e.g., AAPL) and internet connection. Try a different ticker (e.g., GOOG, MSFT).
Plotly Graph Not Rendering: Ensure plotly is installed (pip install plotly) and Streamlit is updated (pip install --upgrade streamlit).
Model Performance: Predictions may vary due to the random initialization of the LSTM. Increase training epochs (e.g., 50) in model_utils.train_model or save/load a trained model using save_model/load_model.
Debug Output: Check debug messages in the Streamlit app (e.g., Downloaded data shape, future_preds shape) to diagnose issues.

If issues persist, share the error message, debug output, and ticker used for further assistance.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please include tests and update documentation as needed.
License
This project is licensed under the MIT License. See the LICENSE file for details.


