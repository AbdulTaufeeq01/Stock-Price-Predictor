# 📈 Stock Price Predictor

A **web-based application** for predicting future stock prices using a **Long Short-Term Memory (LSTM)** neural network implemented in **PyTorch**. The app fetches historical stock data, computes **technical indicators** (Simple Moving Average, Exponential Moving Average, and Relative Strength Index), trains an LSTM model, and visualizes predictions using an interactive **Plotly** graph. Built with **Streamlit** for a clean and intuitive UI.

---

## 🧭 Table of Contents

- [🚀 Features](#-features)
- [🛠 Installation](#-installation)
- [📈 Usage](#-usage)
- [🗂 File Structure](#-file-structure)
- [📦 Dependencies](#-dependencies)
- [⚙️ How It Works](#️-how-it-works)
- [🐞 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Features

- 📉 **Data Retrieval**: Fetches historical stock data from Yahoo Finance using `yfinance`.
- 📊 **Technical Indicators**: Computes 20-day and 50-day **SMA/EMA**, and 14-day **RSI** to enhance model predictions.
- 🧠 **LSTM Model**: Implements a PyTorch-based LSTM for time series forecasting.
- 📈 **Interactive Visualization**: Displays historical and predicted prices with **Plotly** in a dynamic graph.
- 🧪 **User Interface**: Built with **Streamlit** – input a stock ticker, select days to predict, choose to retrain model.
- ✅ **Robust Error Handling**: Catches and informs users of issues like empty data or invalid tickers.

---

## 🛠 Installation

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
```
### 2.**Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
# Activate:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
### 4.  **Run the Application**
```bash
streamlit run app.py
```
## 📈 Usage
- Open the app in your browser (usually http://localhost:8501).
- Enter a valid stock ticker (e.g., AAPL).
- Select number of days to predict (1 to 30).
- Check "Retrain Model" to retrain the LSTM or uncheck to reuse the previous model (if caching is implemented).
- Click "Predict" to start.

## 🔍 Output
- A table showing the predicted closing prices.
- A Plotly line graph with:
- Last 60 days of actual closing prices
- Predicted values for the next n days
- A vertical line marking the start of prediction

## 📊 Example Output Table (7-day prediction)
Day	Predicted Price (USD)
Day 1	219.16
Day 2	218.90
Day 3	218.40
Day 4	217.78
Day 5	217.12
Day 6	216.47
Day 7	215.86

## 🗂 File Structure
```bash
.
├── app.py               # 🚀 Streamlit app script
├── data_utils.py        # 📊 Data handling, indicators, scaling
├── indicators.py        # 🧮 RSI indicator
├── model_utils.py       # 🧠 Model, training, prediction
├── requirements.txt     # 📦 Python dependencies
└── README.md            # 📘 Documentation
```
## 📦 Dependencies
- Python 3.8 or higher
- streamlit>=1.20.0
- yfinance>=0.2.0
- pandas>=1.5.0
- numpy>=1.23.0
- torch>=2.0.0
- scikit-learn>=1.2.0
- plotly>=5.10.0

Install them manually if needed:
```bash
pip install streamlit yfinance pandas numpy torch scikit-learn plotly
```
## ⚙️ How It Works

### 📥 Data Download
- Fetches historical stock data from **Yahoo Finance** using the `yfinance` library.

### 🛠️ Feature Engineering
Calculates the following technical indicators:
- **Simple Moving Averages (SMA):** 20-day and 50-day
- **Exponential Moving Averages (EMA):** 20-day and 50-day
- **Relative Strength Index (RSI):** 14-day

### 📊 Data Scaling
- Applies `MinMaxScaler` to normalize features such as:
  - Close
  - SMA
  - EMA
  - RSI

### 🔁 Sequence Creation
- Uses a **60-day lookback window** to generate input sequences for training the LSTM model.

### 🧠 Model Training
- Trains a **3-layer LSTM model** implemented in PyTorch with:
  - 100 hidden units
  - Dropout for regularization
  - MSE loss for error calculation

### 🔮 Prediction
- Uses a **recursive strategy** to predict stock prices for the next `n` days.

### 🔁 Inverse Transformation
- Converts the scaled predictions back to the original price scale using the same scaler.

### 📈 Visualization
- Combines:
  - Last 60 actual stock prices
  - Next `n` predicted prices
- Displays them on an **interactive Plotly chart** for clear and intuitive comparison.


## 🐞 Troubleshooting
- MultiIndex Errors
- Make sure the DataFrame from yfinance does not have MultiIndex columns. Use .reset_index() if needed.
- Empty DataFrame
Happens with invalid tickers or network issues. Try tickers like AAPL, GOOG, MSFT.
- Plotly Not Rendering
- Ensure plotly is installed:
```bash
pip install plotly
```
- Model Too Inaccurate?
Increase training epochs in train_model() or enhance feature selection.
- Check Debug Logs
Logs will help debug shapes, scaling errors, or prediction failures.

## 🤝 Contributing
- Pull requests are welcome! Here's how to contribute:
- Fork the repository
- Create a new branch
```bash
git checkout -b feature/your-feature
```
- Make your changes
- Commit and push
```bash
git commit -m "Added new feature"
git push origin feature/your-feature
```
- Open a Pull Request
- Please include tests and update the documentation when applicable.
