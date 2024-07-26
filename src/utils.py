# import dependancies
import requests
import pandas as pd


# Function to get daily stock prices
def get_daily_stock_prices(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    # Extract time series data
    time_series = data.get('Time Series (Daily)', {})
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df

# Function to get financial news
def get_financial_news(api_key, tickers):
    # Ensure tickers are properly formatted
    if not tickers:
        raise ValueError("Tickers parameter is required.")
    
    # Construct the URL
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={api_key}'
    
    # Make the request
    response = requests.get(url)
    data = response.json()
    
    # Debug: Print the raw JSON response
    print("Financial News API Response:", data)
    
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