# train_stock.py
import logging
from models.train import create_model, train_model, validate_model
from data.pull import preprocess_data, prepare_datasets

logger = logging.getLogger(__name__)

def train_and_save_model(symbol):
    try:
        scaled_data, scaler, _ = preprocess_data(symbol)
        x_train, y_train, x_val, y_val = prepare_datasets(scaled_data)
        model = create_model()
        train_model(model, x_train, y_train)
        rmse, mae, mape, _, _ = validate_model(model, x_val, y_val, scaler)
        model_path = f'models/{symbol}_prediction.h5'
        model.save(model_path)
        logger.info(f"Model for {symbol} trained and saved successfully. Metrics: RMSE={rmse}, MAE={mae}, MAPE={mape}")
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        raise
