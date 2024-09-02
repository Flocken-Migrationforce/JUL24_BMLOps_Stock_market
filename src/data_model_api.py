from fastapi import FastAPI, HTTPException, Request, Depends
import httpx # to connect jobs between FastAPI instances
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

from models.train import create_model, train_model, validate_model
from models.predict import predict_prices
from data.preprocessing import process_data, preprocess_data
from data.pull import get_daily_stock_prices, create_my_dataset, prepare_datasets
from visualization.visualize import generate_visualizations, create_stock_chart
#
from auth import get_current_user  # Import the authentication dependency

app = FastAPI()


class StockRequest(BaseModel):
    """Model representing a request to predict stock prices."""
    symbol: str
    userid: str  # Include userid in the request to identify the user


@app.post("/process")
async def process(request: Request):
    req_json = await request.json()
    if 'data' not in req_json:
        raise HTTPException(status_code=400, detail="No data provided")

    data = req_json['data']
    try:
        processed_data = process_data(data)
        return {"processed_data": processed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
async def visualize(request: Request):
    req_json = await request.json()
    if 'data' not in req_json:
        raise HTTPException(status_code=400, detail="No data provided")

    data = req_json['data']
    try:
        visualization = generate_visualizations(data)
        return {"visualization": visualization}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/")
async def train_model_endpoint(stock_request: StockRequest, current_user: dict = Depends(get_current_user)):
    """
    Train a stock prediction model for the specified symbol.

    Args:
        stock_request (StockRequest): The stock symbol and user ID for the request.
        current_user (dict): The authticated user data (automatically injected by Depends).

    Returns:
        dict: A message confirming the model was trained and saved.
    """

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


@app.post("/predict/")USER_MANAGEMENT_API_URL = "http://localhost:8001"  # Adjust this URL as needed

async def get_user_data(userid: str, token: str):
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{USER_MANAGEMENT_API_URL}/users/{userid}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch user data")

@app.post("/predict/")
async def predict_stock(stock_request: StockRequest, token: str = Depends(oauth2_scheme)):
    """
    Predict stock prices for the specified symbol.
    Only users with a premium subscription can access this feature.
    """
    symbol = stock_request.symbol.upper()
    userid = stock_request.userid

    # Fetch user data from User Management API
    user = await get_user_data(userid, token)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user['subscription'] != "premium":
        raise HTTPException(status_code=403, detail="Your membership is not premium. Please upgrade to access this feature.")

    try:
        # Assuming train_validate_predict function returns necessary prediction data
        scaled_data, scaler, _ = preprocess_data(symbol)
        model_path = f'models/{symbol}_prediction.h5'
        model = load_model(model_path)
        predicted_prices = predict_prices(model, scaled_data, scaler, prediction_days=7)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "symbol": symbol,
        "predicted_prices": predicted_prices.tolist()
    }



@app.get("/visualize/{symbol}", response_class=HTMLResponse)
async def visualize_stock(request: Request, symbol: str, days: int = 7):
    try:
        scaled_data, scaler, stock_prices_df = preprocess_data(symbol)
        model_path = f'models/{symbol}_prediction.h5'
        model = load_model(model_path)
        predicted_prices = predict_prices(model, scaled_data, scaler, prediction_days=days)
        chart_image = create_stock_chart(stock_prices_df, predicted_prices, symbol)

        return HTMLsites.TemplateResponse("stock_visualization.html", {
            "request": request,
            "symbol": symbol,
            "chart_image": chart_image,
            "prediction_days": days
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("data_model:app", host="0.0.0.0", port=8002, reload=True)