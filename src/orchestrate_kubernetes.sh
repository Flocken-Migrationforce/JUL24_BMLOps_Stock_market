#!/bin/bash

# Check if YOUR_DOCKER_USERNAME is set
if [ -z "$YOUR_DOCKER_USERNAME" ]; then
    echo "Error: YOUR_DOCKER_USERNAME environment variable is not set."
    echo "Please set it by running: export YOUR_DOCKER_USERNAME=your_docker_username"
    exit 1
fi

# Apply the Kubernetes configuration
envsubst < kubernetes-deployment.yaml | kubectl apply -f -

echo "Kubernetes deployment applied successfully."