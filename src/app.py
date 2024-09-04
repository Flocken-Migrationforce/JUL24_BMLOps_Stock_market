
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# List of supported symbols
SUPPORTED_SYMBOLS = ["AAPL", "GOOGL", "EUR/USD", "GOLD"]


import auth

app = FastAPI()
DATA_MODEL_URL = "localhost:8000"


# Setting up HTML directory for displaying HTML Front End
from fastapi.templating import Jinja2Templates
HTMLsites = Jinja2Templates(directory="HTML")

# Mount the static directory for CSS and other static files
app.mount("/static", StaticFiles(directory="HTML"), name="HTML")

# Set up logging configuration
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



##############################################################################################################
# FRONT END
##############################################################################################################

# Endpoints :

@app.get("/")
async def home(request: Request):
    return HTMLsites.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train_page(request: Request):
    return HTMLsites.TemplateResponse("train.html", {"request": request})


@app.get("/predict")
async def predict_page(request: Request):
    return HTMLsites.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict_stock(request: Request):
    form_data = await request.form()
    symbol = form_data.get("symbol")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{DATA_MODEL_URL}/predict/", json={"symbol": symbol})

    result = response.json()
    return HTMLsites.TemplateResponse("predict_result.html", {"request": request, "result": result})


##############################################################################################################


##############################################################################################################
# USER MANAGEMENT
##############################################################################################################


from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# This should be kept secret and not in the code
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    password: str
    subscription: str

class UserInDB(User):
    hashed_password: str


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

# Function to write a new user to the database_users file
def write_user_to_file(user: User, filename="database_users.csv"):
    try:
        with open(filename, 'a') as file:
            file.write(f"{user.username},{user.password},{user.subscription}\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to file: {str(e)}")

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


@app.post("/users/")
async def create_user(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)
    users_db[user.username] = {
        "username": user.username,
        "hashed_password": hashed_password,
        "subscription": user.subscription
    }


    # Update the file
    with open(USERS_FILE, 'a') as file:
        file.write(f"{user.username},{user.password},{user.subscription}\n")

    return {"username": user.username, "subscription": user.subscription}

@app.get("/users/")
async def get_users(current_user: User = Depends(get_current_user)):
    return [{"username": username, "subscription": user["subscription"]}
            for username, user in users_db.items()]


@app.put("/users/{username}")
async def update_user(username: str, user: User, current_user: User = Depends(get_current_user)):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    users_db[username]["subscription"] = user.subscription

    # Update the file
    with open(USERS_FILE, 'w') as file:
        for u_name, u_data in users_db.items():
            file.write(f"{u_name},{u_data['hashed_password']},{u_data['subscription']}\n")

    return {"username": username, "subscription": user.subscription}


@app.delete("/users/{username}")
async def delete_user(username: str, current_user: User = Depends(get_current_user)):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    del users_db[username]

    # Update the file
    with open(USERS_FILE, 'w') as file:
        for u_name, u_data in users_db.items():
            file.write(f"{u_name},{u_data['hashed_password']},{u_data['subscription']}\n")

    return {"username": username, "deleted": True}


def load_users_from_file(filename):
    users_db = {}

    with open(filename, 'r') as file:
        for line in file:
            username, password, subscription = line.strip().split(',')
            users_db[username] = {
                "username": username,
                "hashed_password": pwd_context.hash(password),
                "subscription": subscription
            }
    return users_db

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt




@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/users/")
async def create_user(user: User):
    """
    Create a new user and write it to the file database_users.csv.

    Args:
        user (User): The user to create.

    Returns:
        User: The created user.
    """
    # Check if the user already exists
    if user.username in auth.users_db:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Hash the password before storing
    hashed_password = auth.pwd_context.hash(user.password)
    user_data = {
        "username": user.username,
        "hashed_password": hashed_password,
        "subscription": user.subscription
    }

    # Add user to the in-memory database
    auth.users_db[user.username] = user_data

    # Write user to the file
    write_user_to_file(user)

    return {"username": user.username, "subscription": user.subscription}

@app.get("/users/")
async def get_users():
    """
    Retrieve all registered users.

    Returns:
        dict: A dictionary of users keyed by username.
    """
    return [{"username": username, "subscription": user["subscription"]}
            for username, user in auth.users_db.items()]


##############################################################################################################







# App Endpoints
@app.post("/users/")
async def create_user(user: auth.User):
    """
    Create a new user with the specified name and subscription.

    Args:
        user (User): The user to create.

    Returns:
        User: The created user with a unique ID.
    """
    user_id = auth.get_next_user_id()
    user.userid = user_id
    auth.users_db[user_id] = user
    return user


@app.get("/users/")
async def get_users():
    """
    Retrieve all registered users.

    Returns:
        dict: A dictionary of users keyed by user ID.
    """
    return auth.users_db


@app.put("/users/{userid}")
async def update_user(userid: int, user: auth.User):
    """
    Update the name and subscription type of an existing user.

    Args:
        userid (int): The ID of the user to update.
        user (User): The new user data.

    Returns:
        User: The updated user data.
    """
    if userid not in auth.users_db:
        raise HTTPException(status_code=404, detail="User not found")

    # Update the user with the new data
    auth.users_db[userid].name = user.name
    auth.users_db[userid].subscription = user.subscription
    return auth.users_db[userid]


@app.delete("/users/{userid}")
async def delete_user(userid: int):
    """
    Delete a user by ID.

    Args:
        userid (int): The ID of the user to delete.

    Returns:
        dict: A confirmation message indicating the user was deleted.
    """
    if userid not in auth.users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del auth.users_db[userid]
    return {"userid": userid, "deleted": True}


##############################################################################################################
# DATA AND ML MODEL
##############################################################################################################


from fastapi import FastAPI, HTTPException, Request, Depends
import httpx # to connect jobs between FastAPI instances
from pydantic import BaseModel

from models.train import create_model, train_model, validate_model
from models.predict import predict_prices
from data.pull import get_daily_stock_prices, create_my_dataset, prepare_datasets, process_data, preprocess_data
from visualization.visualize import generate_visualizations, create_stock_chart
#
from auth import get_current_user  # Import the authentication dependency

from keras.models import load_model


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


@app.post("/predict/")
async def predict_stock(stock_request: StockRequest, token: str = Depends(oauth2_scheme)):
    """
    Predict stock prices for the specified symbol.
    Only users with a premium subscription can access this feature.
    """
    symbol = stock_request.symbol.upper()
    userid = stock_request.userid

    # Fetch user data from User Management API
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{DATA_MODEL_URL}/users/{userid}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch user data")

        user = response.json()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.get('subscription') != "premium":
        raise HTTPException(status_code=403,
                            detail="Your membership is not premium. Please upgrade to access this feature.")

    try:
        # Assuming these functions are defined elsewhere in your code
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



# Load users from file
USERS_FILE = "database_users.csv"
users_db = load_users_from_file(USERS_FILE)

# Function to start each FastAPI instance
if __name__ == "__main__":
    import init
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    ##############################################################################################################




# Shutdown logging
@app.on_event("shutdown")
def shutdown_event():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"App was shut down at {current_time}")
