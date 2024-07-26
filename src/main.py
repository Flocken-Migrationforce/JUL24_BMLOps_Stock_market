import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os
import tensorflow as tf
from dotenv import load_dotenv
from utils import get_daily_stock_prices, get_financial_news
#################################################################### API & Load Data & Stock Symbol 
# Load environment variables from API.env file
load_dotenv(dotenv_path='API.env')

# Your API key
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

# Example usage
if __name__ == "__main__":
    symbol = 'AAPL'
    stock_prices_df = get_daily_stock_prices(symbol, api_key)
    print("Stock Prices DataFrame:")
    print(stock_prices_df)

    tickers = 'AAPL'
    try:
        financial_news_df = get_financial_news(api_key, tickers=tickers)
        if financial_news_df.empty:
            print("No financial news available for the specified tickers.")
        else:
            print("Financial News DataFrame:")
            print(financial_news_df)
    except ValueError as e:
        print(e)

