from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import csv
import pandas as pd
import secrets
from main import train_validate_predict
from typing import Optional


app = FastAPI()

# File for storing user data
users_file = 'users.csv'

# Helper functions to manage CSV data
def read_users():
    try:
        return pd.read_csv(users_file).set_index('name').to_dict(orient='index')
    except FileNotFoundError:
        return {}

def write_user(user):
    with open(users_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['userid', 'name', 'password', 'subscription'])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(user)

def authenticate_user(name, password):
    users = read_users()
    user = users.get(name)
    if user and secrets.compare_digest(password, user['password']):
        return user
    else:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

# Pydantic models for requests
class User(BaseModel):
    """Model representing a user in the application."""
    userid: Optional[int] = None
    name: str
    password: str
    subscription: str

class TrainRequest(BaseModel):
    """Model for requesting model training."""
    symbol: str  # Include the stock symbol

class PredictRequest(BaseModel):
    """Model for requesting stock predictions."""
    symbol: str
    name: str
    password: str

@app.post("/users/")
def create_user(user: User):
    """Create a new user with the specified name, password, and subscription."""
    user_id = len(read_users()) + 1
    user.userid = user_id
    user_dict = user.dict()
    write_user(user_dict)
    return user

@app.get("/users/")
def get_users():
    """Retrieve all registered users."""
    return read_users()

@app.post("/train/")
def train_model(train_request: TrainRequest):
    """Train a stock prediction model for the specified symbol."""
    symbol = train_request.symbol.upper()
    if symbol not in ['AAPL', 'META', 'MSFT', 'GTLB']:
        raise HTTPException(status_code=400, detail="Unsupported stock symbol. Choose from AAPL, META, MSFT, GTLB.")
    try:
        predictions_val, y_val, _ = train_validate_predict(symbol=symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Model for {symbol} trained and saved successfully."}

@app.post("/predict/")
def predict_stock(predict_request: PredictRequest):
    """Predict stock prices for the specified symbol and user credentials."""
    user = authenticate_user(predict_request.name, predict_request.password)
    if user['subscription'] != 'premium':
        raise HTTPException(status_code=403, detail="Your membership is not premium. Please upgrade to access this feature.")
    symbol = predict_request.symbol.upper()
    try:
        _, _, predicted_prices = train_validate_predict(symbol=symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "symbol": symbol,
        "predicted_prices": predicted_prices.tolist()
    }
