#delete_this_file.
# trashed Fabian 2408231531
# moved all to utils.py
#
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import get_daily_stock_prices, create_my_dataset


## deprecated in favor of split functions for each elementary task of model action : load_data, train, get_metrics, predict.
## 2408231525 Fabian
## We should also consider integration of a data base, to store :
##      1. User credentials
##      2. data of stock prices pulled to local device
##      3. models (pickled or dumped)
##      4. model performances
##      not storing predictions in data base.
''' deprecated 2408231527 Fabian. Split up the tasks into separate elementary functions : Divide et impera.
# Function to train, validate, and predict stock prices
def train_validate_predict(symbol: str, start_date: str = None, end_date: str = None, interval: str = '1d', prediction_days: int = 7):
    # Define the model path
    model_path = f'models/{symbol}_prediction.h5'

    # Fetch daily stock prices for the specified date range and interval
    stock_prices_df = get_daily_stock_prices(symbol, start_date=start_date, end_date=end_date, interval=interval)

    # Preprocess the data
    df = stock_prices_df['1. open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Prepare the training and validation sets
    train_size = int(len(scaled_data) * 0.8)
    dataset_train = scaled_data[:train_size]
    dataset_val = scaled_data[train_size - 60:]  # Overlapping to ensure LSTM can use previous data

    # Create datasets for LSTM training and validation
    x_train, y_train = create_my_dataset(dataset_train, time_step=60)
    x_val, y_val = create_my_dataset(dataset_val, time_step=60)

    # Reshape data for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    # Check if model exists
    if not os.path.exists(model_path):
        # Define the LSTM model architecture
        model = Sequential()
        model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(x_train, y_train, epochs=50, batch_size=32)

        # Save the model
        model.save(model_path)
        print(f"Model trained and saved at {model_path}")
    else:
        print(f"Model already exists. Loading model from {model_path}")
        model = load_model(model_path)

    # Validate the model
    predictions_val = model.predict(x_val)
    predictions_val = scaler.inverse_transform(predictions_val)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

    # Calculate and print validation accuracy metrics
    rmse = np.sqrt(mean_squared_error(y_val, predictions_val))
    mae = mean_absolute_error(y_val, predictions_val)
    mape = np.mean(np.abs((y_val - predictions_val) / y_val)) * 100

    print(f"Validation RMSE: {rmse}")
    print(f"Validation MAE: {mae}")
    print(f"Validation MAPE: {mape}%")

    # Predict the next few days
    last_60_days = scaled_data[-60:]
    x_predict = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    predicted_prices = []
    for _ in range(prediction_days):
        predicted_price = model.predict(x_predict)
        predicted_prices.append(predicted_price[0, 0])
        x_predict = np.append(x_predict[:, 1:, :], np.reshape(predicted_price, (1, 1, 1)), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    return predictions_val, y_val, predicted_prices
'''