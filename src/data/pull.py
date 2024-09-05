import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def get_daily_stock_prices(symbol, start_date=None, end_date=None, interval='1d'):
    """
    Download historical stock data from Yahoo Finance.
    """
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if stock_data.empty:
            raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")
        stock_data.reset_index(inplace=True)
        if 'Open' in stock_data.columns and '1. open' not in stock_data.columns:
            stock_data.rename(columns={'Open': '1. open'}, inplace=True)
        return stock_data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")


def prepare_datasets(scaled_data, time_step=60):
    if len(scaled_data) < time_step:
        raise ValueError("Not enough data to reshape for the given time step.")
    train_size = int(len(scaled_data) * 0.8)
    dataset_train = scaled_data[:train_size]
    dataset_val = scaled_data[train_size - time_step:]
    x_train, y_train = create_my_dataset(dataset_train, time_step=time_step)
    x_val, y_val = create_my_dataset(dataset_val, time_step=time_step)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    return x_train, y_train, x_val, y_val


def create_model(time_step=60):
    model = Sequential([
        LSTM(units=96, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=96, return_sequences=True),
        Dropout(0.2),
        LSTM(units=96),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def create_my_dataset(dataset, time_step=50):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)


def preprocess_data(symbol, start_date=None, end_date=None, interval='1d'):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    stock_prices_df = get_daily_stock_prices(symbol, start_date=start_date, end_date=end_date, interval=interval)
    if '1. open' not in stock_prices_df.columns:
        stock_prices_df['1. open'] = stock_prices_df['Open']
    df = stock_prices_df['1. open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, stock_prices_df


def validate_model(model, x_val, y_val, scaler):
    predictions_val = model.predict(x_val)
    predictions_val = scaler.inverse_transform(predictions_val)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_val, predictions_val))
    mae = mean_absolute_error(y_val, predictions_val)
    mape = np.mean(np.abs((y_val - predictions_val) / y_val)) * 100
    return rmse, mae, mape, predictions_val, y_val


def predict_prices(model, scaled_data, scaler, prediction_days, time_step=60):
    if len(scaled_data) < time_step:
        raise ValueError("Insufficient data for predictions.")
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