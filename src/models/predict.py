#2408231528 Fabian
import numpy as np

def predict_prices(model, scaled_data, scaler, prediction_days):
    """
    Predicts the price for the next days based on the trained model.

    MAYBE OLD! 2409021143FF

    :param model: choose pickled model
    :param scaled_data:
    :param scaler:
    :param prediction_days: Prediction range in days
    :return: Stock prices array.
    """
    last_60_days = scaled_data[-60:]
    x_predict = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_prices = []
    for _ in range(prediction_days):
        predicted_price = model.predict(x_predict)
        predicted_prices.append(predicted_price[0, 0])
        x_predict = np.append(x_predict[:, 1:, :], np.reshape(predicted_price, (1, 1, 1)), axis=1)
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices
