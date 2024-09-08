from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf


def get_daily_stock_prices(symbol, start_date=None, end_date=None, interval='1d'):
    """ Fetch historical stock data from Yahoo Finance. """
    stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={'Open': '1. open'}, inplace=True)
    return stock_data

def create_my_dataset(dataset, time_step=60):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

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
    

def train_aapl():
    symbol = 'AAPL'
    stock_prices_df = get_daily_stock_prices(symbol)
    df = stock_prices_df['1. open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    x_train, y_train, x_val, y_val = create_my_dataset(scaled_data, time_step=60)
    model = create_model(time_step=60)
    model.fit(x_train, y_train, epochs=50, batch_size=32)
    predictions_val = model.predict(x_val)
    predictions_val = scaler.inverse_transform(predictions_val)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_val, predictions_val))
    print(f'Training completed with RMSE: {rmse}')

