from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pull import get_daily_stock_prices

def process_data(data):
    try:
        cleaned_data = clean_data(data)
        transformed_data = transform_data(cleaned_data)
        return transformed_data
    except Exception as e:
        raise RuntimeError(f"Error processing data: {str(e)}")

def clean_data(data):
    try:
        # Assuming data is a Pandas DataFrame
        df = pd.DataFrame(data)
        df.dropna(inplace=True)  # Example of cleaning: removing NaN values
        return df
    except Exception as e:
        raise RuntimeError(f"Error cleaning data: {str(e)}")

def transform_data(cleaned_data):
    try:
        # Example transformation: Scaling the data
        transformed_data = cleaned_data.copy()
        transformed_data['x'] = transformed_data['x'] * 10  # Example operation
        return transformed_data
    except Exception as e:
        raise RuntimeError(f"Error transforming data: {str(e)}")




def preprocess_data(symbol, start_date=None, end_date=None, interval='1d'):
    stock_prices_df = get_daily_stock_prices(symbol, start_date=start_date, end_date=end_date, interval=interval)
    df = stock_prices_df['1. open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, stock_prices_df