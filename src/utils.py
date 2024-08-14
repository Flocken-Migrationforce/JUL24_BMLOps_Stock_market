import yfinance as yf
import numpy as np
import pandas as pd

def get_daily_stock_prices(symbol, start_date=None, end_date=None):
    # Download historical data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Ensure the DataFrame is returned with a similar format
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={'Open': '1. open'}, inplace=True)
    
    return stock_data

def create_my_dataset(dataset, time_step=50):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)
