

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from visualize import generate_visualizations
from visualization.visualize import create_stock_chart
import os

# from model import preprocess_data, prepare_datasets, create_model, train_model, validate_model, predict_prices
import models.train_model
import models.predict_model
import httpx  # Example for making async HTTP requests, waiting for processes to finish before calling another HTML
from datetime import datetime


# from logger_2408282248 import logger
import logging
import authen
import utils

app = FastAPI()

# Setting up HTML directory for displaying HTML Front End
HTMLsites = Jinja2Templates(directory="HTML")

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
	return HTMLsites.TemplateResponse("index.html", {"request": request})


# App Endpoints
@app.post("/users/")
async def create_user(user: authen.User):
	"""
	Create a new user with the specified name and subscription.

	Args:
		user (User): The user to create.

	Returns:
		User: The created user with a unique ID.
	"""
	user_id = authen.get_next_user_id()
	user.userid = user_id
	authen.users_db[user_id] = user
	return user


@app.get("/users/")
async def get_users():
	"""
	Retrieve all registered users.

	Returns:
		dict: A dictionary of users keyed by user ID.
	"""
	return authen.users_db


@app.put("/users/{userid}")
async def update_user(userid: int, user: authen.User):
	"""
	Update the name and subscription type of an existing user.

	Args:
		userid (int): The ID of the user to update.
		user (User): The new user data.

	Returns:
		User: The updated user data.
	"""
	if userid not in authen.users_db:
		raise HTTPException(status_code=404, detail="User not found")

	# Update the user with the new data
	authen.users_db[userid].name = user.name
	authen.users_db[userid].subscription = user.subscription
	return authen.users_db[userid]


@app.delete("/users/{userid}")
async def delete_user(userid: int):
	"""
	Delete a user by ID.

	Args:
		userid (int): The ID of the user to delete.

	Returns:
		dict: A confirmation message indicating the user was deleted.
	"""
	if userid not in authen.users_db:
		raise HTTPException(status_code=404, detail="User not found")
	del authen.users_db[userid]
	return {"userid": userid, "deleted": True}


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
async def train_model_endpoint(stock_request: utils.StockRequest):
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
async def predict_stock(stock_request: utils.StockRequest):
    symbol = stock_request.symbol.upper()
    userid = stock_request.userid

    user = authen.users_db.get(userid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.subscription != "premium":
        raise HTTPException(status_code=403,
                            detail="Your membership is not premium. Please upgrade to access this feature.")

    try:
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


"""@app.get("/async-data/")
async def get_async_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
"""

# start app when this file is run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# Shutdown logging
@app.on_event("shutdown")
def shutdown_event():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"App was shut down at {current_time}")