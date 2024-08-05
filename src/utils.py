import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Function to get daily stock prices
def get_daily_stock_prices(symbol, api_key, start_date=None, end_date=None):
    # Construct the URL for API request
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    
    # Extract time series data
    time_series = data.get('Time Series (Daily)', {})
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter by date range
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    return df.reset_index().rename(columns={'index': 'timestamp'})

# Function to get financial news
def get_financial_news(api_key, tickers, start_date=None, end_date=None):
    # Convert start_date and end_date to datetime if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Set default values for start_date and end_date
    end_date = end_date or datetime.today()
    start_date = start_date or (end_date - timedelta(days=1000))
    
    # Check that start_date and end_date are datetime objects
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime objects")
    
    # Construct the URL
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&time_from={start_date.strftime("%Y-%m-%d")}&time_to={end_date.strftime("%Y-%m-%d")}&apikey={api_key}'
    
    # Make the request
    response = requests.get(url)
    data = response.json()
    
    # Check for valid response
    if 'feed' not in data:
        if 'Information' in data:
            raise ValueError(data['Information'])
        else:
            raise ValueError('Invalid response from API. Please check your inputs.')
    
    # Extract news data
    news_items = data.get('feed', [])
    
    # Convert to DataFrame
    df = pd.DataFrame(news_items)
    
    return df

# Define dataset for LSTM model with previous 50 days prices
def create_my_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y
