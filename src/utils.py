import yfinance as yf
import numpy as np
import pandas as pd

def get_daily_stock_prices(symbol, start_date=None, end_date=None, interval='1d'):
    """
    Download historical stock data from Yahoo Finance.

    Parameters:
    - symbol (str): The stock symbol to fetch data for.
    - start_date (str): The start date for fetching historical data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for fetching historical data in 'YYYY-MM-DD' format.
    - interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    - pd.DataFrame: A DataFrame containing the historical stock prices.
    """
    # Download historical data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
    
    # Ensure the DataFrame is returned with a similar format
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={'Open': '1. open'}, inplace=True)
    
    return stock_data

def create_my_dataset(dataset, time_step=50):
    """
    Create the dataset for training the LSTM model.

    Parameters:
    - dataset (np.array): The dataset to transform into LSTM-compatible format.
    - time_step (int): The number of time steps to look back for prediction.

    Returns:
    - x (np.array): Input features for LSTM.
    - y (np.array): Target variable.
    """
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)
