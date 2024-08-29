#!/bin/bash

# Check if YOUR_DOCKER_USERNAME is set
if [ -z "$YOUR_DOCKER_USERNAME" ]; then
    echo "Error: YOUR_DOCKER_USERNAME environment variable is not set."
    echo "Please set it by running: export YOUR_DOCKER_USERNAME=your_YOUR_DOCKER_USERNAME"
    exit 1
fi

# Build the Docker image
docker build -t stock-prediction-app .

# Tag the image
docker tag stock-prediction-app:latest $YOUR_DOCKER_USERNAME/stock-prediction-app:latest

# Login to Docker Hub (you might want to use Docker credentials store for automation)
docker login

# Push the image
docker push $YOUR_DOCKER_USERNAME/stock-prediction-app:latest

echo "Docker image built and pushed successfully."
echo "Docker is ready for Kubernetes orchestration."