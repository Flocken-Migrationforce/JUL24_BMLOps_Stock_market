import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from utils import get_daily_stock_prices, create_my_dataset

def train_and_save_model(symbol: str):
    # Define the date range
    start_date = '2023-03-11'
    end_date = '2024-07-10'

    # Fetch stock prices data
    stock_prices_df = get_daily_stock_prices(symbol, start_date=start_date, end_date=end_date)

    # Preprocess the data
    df = stock_prices_df['1. open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df)
    x_train, y_train = create_my_dataset(dataset)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Save the model with the stock symbol in the filename
    model_path = f'models/{symbol}_prediction.h5'
    model.save(model_path)
    print(f"Model saved at {model_path}")

def predict_stock_price(symbol: str):
    # Define the date range
    start_date = '2023-03-11'
    end_date = '2024-07-10'

    # Fetch daily stock prices for the specified date range
    stock_prices_df = get_daily_stock_prices(symbol, start_date=start_date, end_date=end_date)

    # Save the stock prices DataFrame to a CSV file
    stock_prices_df.to_csv(f'data/{symbol}.csv', index=False)

    # Load the stock prices data from the CSV file
    df = pd.read_csv(f'data/{symbol}.csv')
    df = df['1. open'].values.reshape(-1, 1)

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df)
    x_test, y_test = create_my_dataset(dataset)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Load the pre-trained model
    model_path = f'models/{symbol}_prediction.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {symbol} not found at {model_path}")

    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions, y_test
