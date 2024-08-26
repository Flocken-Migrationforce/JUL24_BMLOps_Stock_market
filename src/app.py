from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional


app = FastAPI()

# In-memory database for users
users_db = {}

# Pydantic model for a User
class User(BaseModel):
    """Model representing a user in the application."""
    userid: Optional[int] = None
    name: str
    subscription: str

class StockRequest(BaseModel):
    """Model representing a request to predict stock prices."""
    symbol: str
    userid: int  # Include userid in the request to identify the user

def get_next_user_id():
    """Helper function to generate the next user ID."""
    return max(users_db.keys(), default=0) + 1

@app.post("/users/")
def create_user(user: User):
    """
    Create a new user with the specified name and subscription.
    
    Args:
        user (User): The user to create.
    
    Returns:
        User: The created user with a unique ID.
    """
    user_id = get_next_user_id()
    user.userid = user_id
    users_db[user_id] = user
    return user

@app.get("/users/")
def get_users():
    """
    Retrieve all registered users.
    
    Returns:
        dict: A dictionary of users keyed by user ID.
    """
    return users_db

@app.put("/users/{userid}")
def update_user(userid: int, user: User):
    """
    Update the name and subscription type of an existing user.
    
    Args:
        userid (int): The ID of the user to update.
        user (User): The new user data.
    
    Returns:
        User: The updated user data.
    """
    if userid not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update the user with the new data
    users_db[userid].name = user.name
    users_db[userid].subscription = user.subscription
    return users_db[userid]

@app.delete("/users/{userid}")
def delete_user(userid: int):
    """
    Delete a user by ID.
    
    Args:
        userid (int): The ID of the user to delete.
    
    Returns:
        dict: A confirmation message indicating the user was deleted.
    """
    if userid not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[userid]
    return {"userid": userid, "deleted": True}

@app.post("/train/")
def train_model(stock_request: StockRequest):
    """
    Train a stock prediction model for the specified symbol.
    
    Args:
        stock_request (StockRequest): The stock symbol and user ID for the request.
    
    Returns:
        dict: A message confirming the model was trained and saved.
    """
    symbol = stock_request.symbol.upper()
    if symbol not in ['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F']:
        raise HTTPException(status_code=400, detail="Unsupported stock symbol")

    try:
        # Assuming the last two parameters in `train_validate_predict` are start_date and end_date
        predictions_val, y_val, _ = train_validate_predict(symbol=symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Model for {symbol} trained and saved successfully."}

@app.post("/predict/")
def predict_stock(stock_request: StockRequest):
    """
    Predict stock prices for the specified symbol.
    
    Only users with a premium subscription can access this feature.
    
    Args:
        stock_request (StockRequest): The stock symbol and user ID for the request.
    
    Returns:
        dict: The predicted stock prices.
    
    Raises:
        HTTPException: If the user is not premium or if the user or model is not found.
    """
    symbol = stock_request.symbol.upper()
    userid = stock_request.userid

    user = users_db.get(userid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.subscription != "premium":
        raise HTTPException(status_code=403, detail="Your membership is not premium. Please upgrade to access this feature.")

    try:
        _, _, predicted_prices = train_validate_predict(symbol=symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "symbol": symbol,
        "predicted_prices": predicted_prices.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
