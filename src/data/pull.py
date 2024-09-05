import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

'''
def test_yf_download(symbol):
    try:
        data = yf.download(symbol, start="2023-01-01", end="2023-12-31")
        print(f"Successfully downloaded data for {symbol}. Shape: {data.shape}")
        print(data.head())
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")


# Test with a few symbols
for symbol in ['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F']:
    test_yf_download(symbol)'''


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
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)

        if stock_data.empty:
            raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")

        # Ensure the DataFrame is returned with a similar format
        stock_data.reset_index(inplace=True)
        if 'Open' in stock_data.columns and '1. open' not in stock_data.columns:
            stock_data.rename(columns={'Open': '1. open'}, inplace=True)

        return stock_data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")



def prepare_datasets(scaled_data, time_step=60):
    train_size = int(len(scaled_data) * 0.8)
    dataset_train = scaled_data[:train_size]
    dataset_val = scaled_data[train_size - time_step:]
    x_train, y_train = create_my_dataset(dataset_train, time_step=time_step)
    x_val, y_val = create_my_dataset(dataset_val, time_step=time_step)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    return x_train, y_train, x_val, y_val


def create_model(time_step=60):
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


def create_my_dataset(dataset, time_step=60):
    """
    Create the dataset for training the LSTM model.

    Parameters:
    - dataset (np.array): The dataset to transform into LSTM-compatible format.
    - time_step (int): The number of time steps to look back for prediction.

    Returns:
    - x (np.array): Input features for LSTM.
    - y (np.array): Target variable.
    """
    dataset = np.array(dataset).reshape(-1, 1)
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
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        stock_prices_df = get_daily_stock_prices(symbol, start_date=start_date, end_date=end_date, interval=interval)

        if stock_prices_df.empty:
            raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")

        if '1. open' not in stock_prices_df.columns:
            stock_prices_df['1. open'] = stock_prices_df['Open']

        df = stock_prices_df['1. open'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        return scaled_data, scaler, stock_prices_df
    except Exception as e:
        logger.error(f"Error preprocessing data for {symbol}: {str(e)}")
        raise ValueError(f"Error preprocessing data for {symbol}: {str(e)}")


from datetime import datetime, timedelta


def get_daily_stock_prices(symbol, start_date=None, end_date=None, interval='1d'):
    # Set end_date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Set start_date to 1000 days ago if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')

    # Convert string dates to datetime objects for comparison
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')

    # Ensure start_date is not after end_date
    if start_date_dt > end_date_dt:
        raise ValueError("start_date cannot be after end_date")
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if stock_data.empty:
            raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")
        return stock_data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")


def validate_model(model, x_val, y_val, scaler):
    predictions_val = model.predict(x_val)
    predictions_val = scaler.inverse_transform(predictions_val)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_val, predictions_val))
    mae = mean_absolute_error(y_val, predictions_val)
    mape = np.mean(np.abs((y_val - predictions_val) / y_val)) * 100
    return rmse, mae, mape, predictions_val, y_val

def predict_prices(model, scaled_data, scaler, prediction_days, time_step=60):
    last_time_step = scaled_data[-time_step:]
    x_predict = np.reshape(last_time_step, (1, last_time_step.shape[0], 1))
    predicted_prices = []
    for _ in range(prediction_days):
        predicted_price = model.predict(x_predict)
        predicted_prices.append(predicted_price[0, 0])
        x_predict = np.append(x_predict[:, 1:, :], np.reshape(predicted_price, (1, 1, 1)), axis=1)
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices



if __name__ == '__main__':
    # test_yf_download('AAPL')
    downloaddata = get_daily_stock_prices('AAPL')
    scaled_data, _, _ = preprocess_data('AAPL')
    prepare_datasets(scaled_data)

    print("wait")
    model = create_model()
    train_model(model, x_train, y_train)
    rmse, mae, mape, _, _ = validate_model(model, x_val, y_val, scaler)
    model_path = f'models/AAPL_prediction.h5'
    model.save(model_path)
    print("wait")
