# 2408011154
# Interface for interactions with our model
# implemented with FastAPI

from fastapi import FastAPI
# from app import auth # throwed error 2408011303

# Import the app functions from our endpoint scripts:
from auth import auth_app
from airflow_to_FastAPI import airflow_app
from shared_database import database_app
from message_queue import message_queue_app
from docker_tasks import docker_tasks_app
from dashboard import dash_app
from model_interactions_PLACEHOLDER import model_app

app = FastAPI()

# Mounting Routers: API instances (in Docker containers) from our endpoint scripts:
app.mount("/auth", auth_app) # include the auth router
app.mount("/airflow", airflow_app)
app.mount("/database", database_app)
app.mount("/message_queue", message_queue_app)
app.mount("/docker_tasks", docker_tasks_app)
app.mount("/dashboard", dash_app.server)
app.mount("/model", model_app)



# Endpoints for FastAPI from not a script file here :
@app.get("/")
async def root():
    return {"message": "Hello World"}

'''
## already mounted. 2408011311

# Include the auth router:
import auth
app.include_router(auth.app, prefix="/auth")



# Build connection layer between Airflow and FastAPI
import airflow_to_FastAPI


# Configure dashboard:
from app.dashboard import dash_app 
app.mount("/dashboard", dash_app.server)
'''
