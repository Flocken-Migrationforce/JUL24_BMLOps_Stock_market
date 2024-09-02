import pytest
import numpy as np
from model import preprocess_data, prepare_datasets, create_model, train_model, validate_model, predict_prices
from unittest.mock import patch, MagicMock
from keras.models import Sequential

def test_preprocess_data(mocker):
    mock_get_daily_stock_prices = mocker.patch('model.get_daily_stock_prices', return_value={"1. open": [100, 200, 300]})
    scaled_data, scaler, stock_prices_df = preprocess_data("AAPL")
    assert scaled_data.shape == (3, 1)
    assert stock_prices_df is not None

def test_prepare_datasets():
    scaled_data = np.array([[i] for i in range(1000)])
    x_train, y_train, x_val, y_val = prepare_datasets(scaled_data)
    assert x_train.shape[0] > 0
    assert y_train.shape[0] > 0
    assert x_val.shape[0] > 0
    assert y_val.shape[0] > 0

def test_create_model():
    model = create_model()
    assert isinstance(model, Sequential)

def test_train_model(mocker):
    mock_model = MagicMock()
    x_train = np.random.rand(100, 60, 1)
    y_train = np.random.rand(100, 1)
    train_model(mock_model, x_train, y_train)
    mock_model.fit.assert_called_once()

def test_validate_model(mocker):
    mock_model = MagicMock()
    x_val = np.random.rand(20, 60, 1)
    y_val = np.random.rand(20, 1)
    mock_predict = mocker.patch.object(mock_model, 'predict', return_value=np.random.rand(20, 1))
    mock_scaler = MagicMock()
    rmse, mae, mape, predictions_val, y_val = validate_model(mock_model, x_val, y_val, mock_scaler)
    assert rmse >= 0
    assert mae >= 0
    assert mape >= 0

def test_predict_prices(mocker):
    mock_model = MagicMock()
    mock_scaler = MagicMock()
    scaled_data = np.random.rand(100, 1)
    prediction_days = 7
    predictions = predict_prices(mock_model, scaled_data, mock_scaler, prediction_days)
    assert predictions.shape == (prediction_days, 1)
