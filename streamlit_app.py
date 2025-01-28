import subprocess
import sys

# Function to install required packages
def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# List of required packages
required_packages = ['yfinance', 'pandas', 'numpy', 'scikit-learn', 'streamlit', 'matplotlib']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)



import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt


# Step 1: Fetch Stock Data for Top 500 Companies
def get_top_companies():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].tolist()[:499]

def fetch_stock_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Step 2: Preprocess Data
def preprocess_data(stock_data, ticker):
    if ticker not in stock_data:
        return None

    df = stock_data[ticker].copy()
    if df.empty or len(df) < 200:
        return None

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)

    # Calculate returns
    df['Return_1D'] = df['Close'].pct_change(1)
    df['Return_1M'] = df['Close'].pct_change(21)
    df['Return_1Y'] = df['Close'].pct_change(252)
    df.dropna(inplace=True)

    return df if not df.empty else None

# Step 3: Run Prediction Model
def predict_stock(df):
    model = LinearRegression()
    X = df[['SMA_50', 'SMA_200']]
    
    # Fit model and predict returns
    df['Predicted_Return_1D'] = model.fit(X, df['Return_1D']).predict(X)
    df['Predicted_Return_1M'] = model.fit(X, df['Return_1M']).predict(X)
    df['Predicted_Return_1Y'] = model.fit(X, df['Return_1Y']).predict(X)

    return df

# Step 4: Recommend Top 3 Stocks for each time horizon
def recommend_top_stocks(stock_data, tickers):
    predictions = {ticker: [] for ticker in tickers}

    for ticker in tickers:
        df = preprocess_data(stock_data, ticker)
        if df is not None:
            df = predict_stock(df)
            predictions[ticker] = [
                df['Predicted_Return_1D'].iloc[-1],
                df['Predicted_Return_1M'].iloc[-1],
                df['Predicted_Return_1Y'].iloc[-1]
            ]

    # Filter out tickers without valid predictions
    valid_predictions = {ticker: pred for ticker, pred in predictions.items() if len(pred) == 3}

    # Sort predictions and get top 3 for each horizon
    top_stocks = {
        '1D': sorted(valid_predictions.items(), key=lambda x: x[1][0], reverse=True)[:3],
        '1M': sorted(valid_predictions.items(), key=lambda x: x[1][1], reverse=True)[:3],
        '1Y': sorted(valid_predictions.items(), key=lambda x: x[1][2], reverse=True)[:3]
    }

    # Prepare results with stock names
    top_stocks_named = {
        period: [(stock, valid_predictions[stock]) for stock, _ in stocks]
        for period, stocks in top_stocks.items()
    }

    return top_stocks_named

# Streamlit Main Function
def main():
    st.set_page_config(page_title="Stock Analysis and Recommendations", layout="wide")
    st.title("📈 Stock Analysis and Recommendations")
    st.sidebar.header("Settings")

    tickers = get_top_companies()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)

    if st.sidebar.button("Fetch Stock Data"):
        with st.spinner("Fetching stock data..."):
            stock_data = fetch_stock_data(tickers, start_date, end_date)

        st.success("Data fetched successfully! Analyzing stocks...")
        top_stocks = recommend_top_stocks(stock_data, tickers)

        for period, stocks in top_stocks.items():
            st.subheader(f"🌟 Top 3 Recommended Stocks for {period} ROI:")
            for stock, predicted_returns in stocks:
                st.write(f"**{stock}:** Predicted Return = {predicted_returns[0]:.4f} (1D), "
                         f"{predicted_returns[1]:.4f} (1M), {predicted_returns[2]:.4f} (1Y)")
                
                # Plotting stock prices
                df = preprocess_data(stock_data, stock)
                if df is not None:
                    plt.figure(figsize=(10, 5))
                    plt.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=2)
                    plt.plot(df.index, df['SMA_50'], label='50-Day SMA', color='orange', linestyle='--', linewidth=2)
                    plt.plot(df.index, df['SMA_200'], label='200-Day SMA', color='red', linestyle='--', linewidth=2)
                    plt.title(f"{stock} Stock Prices with SMA", fontsize=16)
                    plt.xlabel("Date", fontsize=12)
                    plt.ylabel("Price", fontsize=12)
                    plt.legend()
                    plt.grid()
                    st.pyplot(plt)

if __name__ == "__main__":
    main()
