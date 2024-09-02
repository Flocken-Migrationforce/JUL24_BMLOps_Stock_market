
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import httpx  # Example for making async HTTP requests, waiting for processes to finish before calling another HTML
from datetime import datetime
import os
import multiprocessing

# from model import preprocess_data, prepare_datasets, create_model, train_model, validate_model, predict_prices
import models.train
import models.predict
from visualization.visualize import generate_visualizations, create_stock_chart



# from logger_2408282248 import logger
import logging
import auth

from auth import get_current_user  # Import the authentication dependency

app = FastAPI()

##deprecatedFor app.mount(HTML) 2408302238start
# Setting up HTML directory for displaying HTML Front End
from fastapi.templating import Jinja2Templates
HTMLsites = Jinja2Templates(directory="HTML")
##deprecatedFor app.mount(HTML) 2408302238end

# Mount the static directory for CSS and other static files
app.mount("/static", StaticFiles(directory="HTML"), name="HTML")

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


'''
##deprecated2408302239 for app.mount in FastAPI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLsites.TemplateResponse("index.html", {"request": request})
'''

@app.get("/")
async def home(request: Request):
    return HTMLsites.TemplateResponse("index.html", {"request": request})
    # return FileResponse("HTML/index.html")  # not dynamic HTML.


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




"""@app.get("/async-data/")
async def get_async_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
"""

# Function to start each FastAPI instance
def start_app(module, port):
    import uvicorn
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    # Start all FastAPI instances
    processes = [
        multiprocessing.Process(target=start_app, args=("app", 8000)),
        multiprocessing.Process(target=start_app, args=("user_management_api", 8001)),
        multiprocessing.Process(target=start_app, args=("backend_api", 8002)),
        multiprocessing.Process(target=start_app, args=("data_model_api", 8003)),
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


'''deprecated2409021413 when 4 FastAPI instances were introduced, not only 1.
# start app when this file is run
if __name__ == "__main__":
    import init
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
'''

# Shutdown logging
@app.on_event("shutdown")
def shutdown_event():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"App was shut down at {current_time}")
