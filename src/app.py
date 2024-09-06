from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import sys

from starlette.responses import JSONResponse

## Set working directory for this script
# # allows to run "python src/app.py and put src as the working directory, so that all normally works.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory (src folder)
os.chdir(script_dir)
##

# List of supported symbols
SUPPORTED_SYMBOLS = ["AAPL", "GOOGL", "GTLE", "META", "MSFT"]


import auth
from prometheus_fastapi_instrumentator import Instrumentator # for premetheus

app = FastAPI()
DATA_MODEL_URL = "localhost:8000"

# Instrumentation
Instrumentator().instrument(app).expose(app)  # to be exposed in prometheus 
# In-memory database for users
users_db = {}

# Setting up HTML directory for displaying HTML Front End
from fastapi.templating import Jinja2Templates
HTMLsites = Jinja2Templates(directory="HTML")

# Mount the static directory for CSS and other static files
app.mount("/static", StaticFiles(directory="HTML"), name="HTML")

# Set up logging configuration
import logging

### LOGGING
# Get the current working directory and define the log directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, '..', 'logs')
os.makedirs(log_dir, exist_ok=True) #  Ensure the log directory exists
# Set up logging file 'app.log':
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log') # creates the log file 'app.log' in this directory 'src'
logger = logging.getLogger(__name__)

# Set up logging filter
class MessageFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("oneDNN custom operations are on")

# Apply the filter to the TensorFlow logger
tf_logger = logging.getLogger('tensorflow')
tf_logger.addFilter(MessageFilter())

# ignore tensorflow warning of using oneDNN custom operations for boosting calculation:
# I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
## DOESNT WORK:
# import warnings, re
# warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow') # ignores all warnings in general from tensorflow.
# warnings.filterwarnings('ignore', message='.*oneDNN custom operations are on.*') # ignore specifically the encountered oneDNN enabled warning.
# warnings.filterwarnings('ignore', message='oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.')
# warnings.filterwarnings('ignore', message=re.compile(r'oneDNN custom operations are on.*floating-point round-off errors.*TF_ENABLE_ONEDNN_OPTS=0')) # DOESN'T WORK for tensorflow INFO I message. Can't use import warnings to cope with that information pop up.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

###

'''more advanced logging option :
# idea Fabian 2409041140
from logging.handlers import RotatingFileHandler

# Create handlers
c_handler = logging.StreamHandler()  # Console handler
f_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=5)  # File handler

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Set the logging level
logger.setLevel(logging.INFO)

'''



##############################################################################################################
# FRONT END
##############################################################################################################

# Endpoints :

@app.get("/")
async def home(request: Request):
    return HTMLsites.TemplateResponse("index.html", {"request": request})

from pydantic import BaseModel

class StockRequest(BaseModel):
    """Model representing a request to predict stock prices."""
    symbol: str
    userid: int  # Include userid in the request to identify the user

from typing import Literal

class StockRequest(BaseModel):
    symbol: Literal['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F']

@app.post("/train/{stocksymbol}")
async def train_model_endpoint(stock_request: StockRequest):
    symbol = stock_request.symbol.upper()
    if symbol not in ['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F']:
        raise HTTPException(status_code=400, detail="Unsupported stock symbol. Predictions are currently only" \
                                                    "allowed for AAPL, GOOGL, EURUSD=X and GC=F.")

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


def train_and_save_model(model, x_train, y_train, x_val, y_val, scaler, symbol, model_path):
    try:
        train_model(model, x_train, y_train)
        rmse, mae, mape, _, _ = validate_model(model, x_val, y_val, scaler)
        model.save(model_path)
        logger.info(f"Model for {symbol} trained and saved successfully.")
        logger.info(f"Metrics for {symbol}: RMSE={rmse}, MAE={mae}, MAPE={mape}")
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")



##############################################################################################################


##############################################################################################################
# USER MANAGEMENT
##############################################################################################################


from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import csv
import auth
from auth import User, UserInDB, UserCreate, TokenData, Token, get_current_user, authenticate_user, create_access_token, get_password_hash, get_next_user_id, users_db, load_users_from_csv, add_user_to_csv

# JWT configueration should be kept secret and not in the code

# Load users from file
USERS_FILE = "database_users.csv"
users_db = load_users_from_csv(USERS_FILE)



@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    print(f"Attempting to authenticate user: {form_data.username}")
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        print(f"Authentication failed for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    print(f"Authentication successful for user: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(
        key=user.key,
        userid=get_next_user_id(USERS_FILE),
        username=user.username,
        full_name=user.full_name,
        email=user.email,
        hashed_password=hashed_password,
        subscription=user.subscription
    )
    add_user_to_csv(USERS_FILE, 'database_passwords.csv', new_user)
    users_db[new_user.username] = new_user
    return new_user


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.get("/users/")
async def get_users(current_user: User = Depends(get_current_user)):
    """
    Retrieve all registered users.

    Returns:
        dict: A dictionary of users keyed by username.
    """
    return [{"username": username, "subscription": user.subscription}
            for username, user in users_db.items()]

@app.put("/users/{username}", response_model=User)
async def update_user(username: str, user: User, current_user: User = Depends(get_current_user)):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    users_db[username]["subscription"] = user.subscription

    # Update the file
    with open(USERS_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['userid', 'username', 'full_name', 'email', 'hashed_password', 'subscription'])
        writer.writeheader()
        for u_name, u_data in users_db.items():
            writer.writerow(u_data.dict())

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

# writes the user's information to the CSV file database_users.csv
def write_user_to_file(user):
    """
    Write user information to the database_users.csv file.

    Args:
        user (User): The user object containing user information.
    """
    USERS_FILE = "database_users.csv"
    with open(USERS_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            user.username,  # 'key' in CSV
            user.userid,
            user.username,
            user.full_name,
            user.email,
            user.hashed_password,
            user.subscription
        ])



##############################################################################################################






##############################################################################################################
# DATA AND ML MODEL
##############################################################################################################


from fastapi import FastAPI, HTTPException, Request, Depends
import httpx # to connect jobs between FastAPI instances
from pydantic import BaseModel

from models.train import create_model, train_model, validate_model
from models.predict import predict_prices
from data.pull import get_daily_stock_prices, create_my_dataset, prepare_datasets, preprocess_data
from visualization.visualize import generate_visualizations, create_stock_chart
#
from auth import get_current_user  # Import the authentication dependency

from tensorflow.keras.models import load_model



class StockRequest(BaseModel):
    symbol: Literal['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F']
    prediction_days: int = 7  # Default to 7 days, but allow user to specify

class PredictionRequest(BaseModel):
    prediction_days: int = 7

@app.post("/predict/{stocksymbol}")
async def predict_stock(
    stocksymbol: Literal['AAPL', 'GOOGL', 'EURUSD=X', 'GC=F'],
    prediction_request: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Handle stock prediction requests for a specific stock symbol.
    This function serves both API and web form submissions.
    Only users with a premium subscription can access this feature.
    """
    if current_user.subscription != "premium":
        raise HTTPException(
            status_code=403,
            detail="Your membership is not premium. Please upgrade to access this feature."
        )

    prediction_days = prediction_request.prediction_days

    try:
        # Load the pre-trained model
        model_path = f'models/{stocksymbol}_prediction.h5'
        model = load_model(model_path)

        # Preprocess the data
        try:
            scaled_data, scaler, _ = preprocess_data(stocksymbol)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Make predictions
        predicted_prices = predict_prices(model, scaled_data, scaler, prediction_days)

        # Convert predictions to a list and round to 2 decimal places
        predicted_prices_list = [round(float(price), 2) for price in predicted_prices.flatten()]

        # Prepare the response
        response = {
            "symbol": stocksymbol,
            "prediction_days": prediction_days,
            "predicted_prices": predicted_prices_list
        }

        return JSONResponse(content=response)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model for {stocksymbol} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



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
        current_user (dict): The authenticated user data (automatically injected by Depends).

    Returns:
        dict: A message confirming the model was trained and saved.
    """
    symbol = stock_request.symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
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

#######################################################################################
## Functions for Monitoring with Prometheus and Grafana and Alertmanager:
#######################################################################################
import subprocess

def start_grafana_port_forward():
    try:
        process = subprocess.Popen(
            ["kubectl", "port-forward", "service/grafana", "3000:80"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started port-forwarding for Grafana on port 3000.")
        return process
    except Exception as e:
        print(f"Failed to start port-forwarding: {e}")
        return None


'''
def start_prometheus():
    try:
        # Start Prometheus using subprocess
        subprocess.Popen(
            ["./prometheus.exe", "--config.file=prometheus.yml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        print("Prometheus has started successfully.")
    except Exception as e:
        print(f"Failed to start Prometheus: {e}")
        '''


def start_prometheus():
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to prometheus.yml
    prometheus_yml_path = os.path.join(current_dir, 'monitoring', 'prometheus.yml')

    # Ensure the path is absolute
    prometheus_yml_path = os.path.abspath(prometheus_yml_path)

    # Docker command to run Prometheus
    docker_command = [
        "docker", "run", "-d",  # Run in detached mode
        "-p", "9090:9090",
        "-v", f"{prometheus_yml_path}:/monitoring/prometheus.yml",
        "prom/prometheus"
    ]

    try:
        subprocess.run(docker_command, check=True)
        print("Prometheus started successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Prometheus: {e}")


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
    else:
        print(stdout.decode())

# Your existing start_prometheus() function remains unchanged

def start_grafana():
    docker_command = [
        "docker", "run", "-d",
        "-p", "3000:3000",
        "grafana/grafana"
    ]
    try:
        subprocess.run(docker_command, check=True)
        print("Grafana started successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Grafana: {e}")


#######################################################################################




##############################################################################################################
##############################################################################################################

'''stopit2409061518Fabian
# Function to start each FastAPI instance
if __name__ == "__main__":
    print("Starting the Stock Prediction App ...")
    # import init
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    #
    # from time import sleep
    # sleep(5)  # Give the app some time to start
    # # start monitoring Prometheus & Grafana & Alertmanager :
    # start_prometheus()
    # start_grafana_port_forward()
    #
    # print("Stock Prediction App started successfully! Ready for interactions.")
    # print("Access Prometheus at http://localhost:9090")
    # print("Access Grafana at http://localhost:3000")
    #
    # # Keep the main thread running
    # try:
    #     while True:
    #         sleep(1)
    # except KeyboardInterrupt:
    #     print("Shutting down...")
##############################################################################################################

'''

# Shutdown logging
@app.on_event("shutdown")
def shutdown_event():
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"App was shut down at {current_time}")
