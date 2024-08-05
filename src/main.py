import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from dotenv import load_dotenv
from utils import get_daily_stock_prices, get_financial_news, create_my_dataset

# Load environment variables from API.env file
load_dotenv(dotenv_path='API.env')

# Your API key from environment variables
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

# Define your stock symbol and date range
symbol = 'AAPL'
start_date = '2023-03-11'
end_date = '2024-07-10'

# Fetch daily stock prices for the specified date range
stock_prices_df = get_daily_stock_prices(symbol, api_key, start_date=start_date, end_date=end_date)
print("Stock Prices DataFrame:")
print(stock_prices_df)

# Save the stock prices DataFrame to a CSV file
stock_prices_df.to_csv(f'data/{symbol}.csv', index=False)

# Load financial news data (Note: financial news data fetching not included in this example)
# financial_news_df = get_financial_news(api_key, tickers=symbol, start_date=start_date, end_date=end_date)
# financial_news_df.to_csv(f'data/{symbol}_news.csv', index=False)  # Save financial news data

#################################################################### Preprocessing

# Load the stock prices data from the CSV file
df = pd.read_csv(r'data/AAPL.csv')

# Display the first few rows of the DataFrame
df.head()

# Extract the '1. open' column as a NumPy array and reshape it
df = df['1. open'].values
df = df.reshape(-1, 1)
print(df.shape)
print(df[:7])

#################################################################### Model Preparation

# Split the data into training and testing sets
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8)-50:])
print(dataset_train.shape)
print(dataset_test.shape)

# Scale the data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_train[:7]

dataset_test = scaler.transform(dataset_test)
dataset_test[:7]

################################################################ LSTM Model

# Create datasets for LSTM training
x_train, y_train = create_my_dataset(dataset_train)
x_test, y_test = create_my_dataset(dataset_test)

# Reshape data for LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_train.shape)
print(x_test.shape)

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.summary()

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model if it does not already exist
if not os.path.exists(r'models/stock_prediction.h5'):
    model.fit(x_train, y_train, epochs=50, batch_size=32)
    model.save(r'models/stock_prediction.h5')

# Load the pre-trained model
model = load_model(r'models/stock_prediction.h5')

# Make predictions and inverse transform them to original scale
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the original and predicted stock prices
fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(df, color='red', label='Original Stock Price')
ax.plot(range(len(y_train)+50, len(y_train)+50+len(predictions)), predictions, color='blue', label='Predicted')
plt.legend()
plt.show()

# Inverse transform y_test and plot it with predictions--> as data was scaled before 
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(y_test_scaled, color='red', label='True Price of Testing Set')
plt.plot(predictions, color='blue', label='Predicted')
plt.legend()
plt.show()
