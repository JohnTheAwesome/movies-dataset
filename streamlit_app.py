import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import streamlit as st

# Set up the page title and description
st.set_page_config(page_title="Stock Analysis", page_icon="📈")
st.title("📈 Stock Analysis App")
st.write(
    """
    This app fetches stock data for the top companies in the S&P 500 and analyzes their performance.
    You can select the number of companies to analyze and view the top recommended stocks based on predicted returns.
    """
)

# Step 1: Fetch Stock Data for Top Companies
def get_top_companies():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].tolist()  # Get all S&P 500 tickers

@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

# Step 2: Preprocess Data
def preprocess_data(stock_data, ticker):
    if ticker not in stock_data:
        return None  # Skip if ticker data is missing

    df = stock_data[ticker].copy()
    if df.empty or len(df) < 200:
        return None

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)

    df['Return_1D'] = df['Close'].pct_change(1)
    df['Return_1M'] = df['Close'].pct_change(21)
    df['Return_1Y'] = df['Close'].pct_change(252)

    df.dropna(inplace=True)

    if df.empty:
        return None

    return df

# Step 3: Run Prediction Model
def predict_stock(df):
    model = LinearRegression()
    X = df[['SMA_50', 'SMA_200']]
    
    df['Predicted_Return_1D'] = model.fit(X, df['Return_1D']).predict(X)
    df['Predicted_Return_1M'] = model.fit(X, df['Return_1M']).predict(X)
    df['Predicted_Return_1Y'] = model.fit(X, df['Return_1Y']).predict(X)

    return df

# Step 4: Recommend Top 3 Stocks for each time horizon
def recommend_top_stocks(stock_data, tickers):
    predictions_1D = {}
    predictions_1M = {}
    predictions_1Y = {}

    for ticker in tickers:
        df = preprocess_data(stock_data, ticker)
        if df is not None:
            df = predict_stock(df)
            predictions_1D[ticker] = df['Predicted_Return_1D'].iloc[-1]
            predictions_1M[ticker] = df['Predicted_Return_1M'].iloc[-1]
            predictions_1Y[ticker] = df['Predicted_Return_1Y'].iloc[-1]

    top_1D = sorted(predictions_1D.items(), key=lambda x: x[1], reverse=True)[:3]
    top_1M = sorted(predictions_1M.items(), key=lambda x: x[1], reverse=True)[:3]
    top_1Y = sorted(predictions_1Y.items(), key=lambda x: x[1], reverse=True)[:3]

    return top_1D, top_1M, top_1Y

# Main Function
def main():
    tickers = get_top_companies()

    # Date selection
    end_date = datetime.now()
    start_date = st.date_input("Select start date", end_date - timedelta(days=365 * 2))
    
    # Fetch stock data
    st.write("Fetching stock data...")
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Analyze stocks
    if st.button("Analyze Stocks"):
        st.write("Analyzing stocks...")
        top_1D, top_1M, top_1Y = recommend_top_stocks(stock_data, tickers)

        # Display results
        st.subheader("Top 3 Recommended Stocks for 1 Day ROI:")
        for stock, predicted_return in top_1D:
            st.write(f"{stock}: Predicted Return = {predicted_return:.4f}")

        st.subheader("Top 3 Recommended Stocks for 1 Month ROI:")
        for stock, predicted_return in top_1M:
            st.write(f"{stock}: Predicted Return = {predicted_return:.4f}")

        st.subheader("Top 3 Recommended Stocks for 1 Year ROI:")
        for stock, predicted_return in top_1Y:
            st.write(f"{stock}: Predicted Return = {predicted_return:.4f}")

if __name__ == "__main__":
    main()
