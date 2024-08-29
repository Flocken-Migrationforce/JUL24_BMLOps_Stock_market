#!/bin/bash

# Set up environment variables
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

# Install Python dependencies
pip install -r app/requirements.txt

# Stop any running Docker containers
docker-compose down

# Initialize Airflow
docker-compose up airflow-init

# Start Docker containers
docker-compose up -d

# Start FastAPI application
cd app
uvicorn main:app --reload &
cd ..

# Display logs (optional)
docker-compose logs -f webserver