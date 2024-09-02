from fastapi import FastAPI
import utils
from fastapi import FastAPI, HTTPException, Request, Depends

import models.train_model
import models.predict_model
from visualize import generate_visualizations
from auth import get_current_user  # Import the authentication dependency

app = FastAPI()


@app.post("/process")
async def process(request: Request):
    req_json = await request.json()
    if 'data' not in req_json:
        raise HTTPException(status_code=400, detail="No data provided")

    data = req_json['data']
    try:
        processed_data = utils.process_data(data)
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
async def train_model_endpoint(stock_request: utils.StockRequest, current_user: dict = Depends(get_current_user)):
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
        scaled_data, scaler, _ = utils.preprocess_data(symbol)
        x_train, y_train, x_val, y_val = utils.prepare_datasets(scaled_data)
        model = utils.create_model()
        utils.train_model(model, x_train, y_train)
        rmse, mae, mape, _, _ = utils.validate_model(model, x_val, y_val, scaler)
        model_path = f'models/{symbol}_prediction.h5'
        model.save(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Model for {symbol} trained and saved successfully.",
            "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape}}


@app.post("/predict/")
async def predict_stock(stock_request: utils.StockRequest, current_user: dict = Depends(get_current_user)):
    """
    Predict stock prices for the specified symbol.

    Only users with a premium subscription can access this feature.

    Args:
        stock_request (StockRequest): The stock symbol and user ID for the request.
        current_user (dict): The authticated user data (automatically injected by Depends).

    Returns:
        dict: The predicted stock prices.

    Raises:
        HTTPException: If the user is not premium or if the user or model is not found.
    """


    symbol = stock_request.symbol.upper()
    userid = stock_request.userid

    # Retrieve the user data from the in-memory database
    user = auth.users_db.get(userid)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.subscription != "premium":
        raise HTTPException(status_code=403,
                            detail="Your membership is not premium. Please upgrade to access this feature.")

    try:
        # Assuming train_validate_predict function returns necessary prediction data
        scaled_data, scaler, _ = utils.preprocess_data(symbol)
        model_path = f'models/{symbol}_prediction.h5'
        model = utils.load_model(model_path)
        predicted_prices = utils.predict_prices(model, scaled_data, scaler, prediction_days=7)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "symbol": symbol,
        "predicted_prices": predicted_prices.tolist()
    }


@app.get("/visualize/{symbol}", response_class=HTMLResponse)
async def visualize_stock(request: Request, symbol: str, days: int = 7):
    try:
        scaled_data, scaler, stock_prices_df = utils.preprocess_data(symbol)
        model_path = f'models/{symbol}_prediction.h5'
        model = utils.load_model(model_path)
        predicted_prices = utils.predict_prices(model, scaled_data, scaler, prediction_days=days)
        chart_image = utils.create_stock_chart(stock_prices_df, predicted_prices, symbol)

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