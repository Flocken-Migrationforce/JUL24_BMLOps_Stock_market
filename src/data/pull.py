import yfinance as yf
import numpy as np

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


def prepare_datasets(scaled_data, time_step=60):
    train_size = int(len(scaled_data) * 0.8)
    dataset_train = scaled_data[:train_size]
    dataset_val = scaled_data[train_size - time_step:]
    x_train, y_train = create_my_dataset(dataset_train, time_step=time_step)
    x_val, y_val = create_my_dataset(dataset_val, time_step=time_step)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    return x_train, y_train, x_val, y_val


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


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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

