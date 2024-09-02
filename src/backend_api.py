# backend_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import data

app = FastAPI()

class StockRequest(BaseModel):
    """Model representing a request to predict stock prices."""
    symbol: str
    userid: int  # Include userid in the request to identify the user


@app.post("/train/")
async def train_model_endpoint(stock_request: StockRequest):
    symbol = stock_request.symbol.upper()
    if symbol not in ['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F']:
        raise HTTPException(status_code=400, detail="Unsupported stock symbol")

    try:
        scaled_data, scaler, _ = preprocess_data(symbol)
        x_train, y_train, x_val, y_val = prepare_datasets(scaled_data)
        model = create_model()
        train_model(model, x_train, y_train)
        rmse, mae, mape, _, _ = validate_model(model, x_val, y_val, scaler)
        model_path = f'models/{symbol}_prediction.h5'
        model.save(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Model for {symbol} trained and saved successfully.",
            "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape}}

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str):
    # Implement logic to fetch stock data
    pass

# Add more endpoints as needed

if __name__ == '__main__':
	import uvicorn
	uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
	# uvicorn backend:app --port 8001