import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from app import app

# Create a TestClient instance to make requests to the FastAPI app
client = TestClient(app)

def test_create_user():
    response = client.post("/users/", json={"name": "Alice", "subscription": "premium"})
    assert response.status_code == 200
    data = response.json()
    assert "userid" in data
    assert data["name"] == "Alice"
    assert data["subscription"] == "premium"

def test_get_users():
    response = client.get("/users/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_update_user():
    # Create a user first
    response = client.post("/users/", json={"name": "Bob", "subscription": "basic"})
    assert response.status_code == 200
    user_id = response.json()["userid"]

    # Update the user's name and subscription
    response = client.put(f"/users/{user_id}", json={"name": "Bob", "subscription": "premium"})
    assert response.status_code == 200
    data = response.json()
    assert data["subscription"] == "premium"
    assert data["name"] == "Bob"

def test_delete_user():
    # Create a user first
    response = client.post("/users/", json={"name": "Charlie", "subscription": "premium"})
    assert response.status_code == 200
    user_id = response.json()["userid"]

    # Delete the user
    response = client.delete(f"/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data == {"userid": user_id, "deleted": True}

def test_train_model():
    # Create a premium user first
    response = client.post("/users/", json={"name": "Dave", "subscription": "premium"})
    assert response.status_code == 200
    user_id = response.json()["userid"]

    # Train a model for a supported stock symbol
    response = client.post("/train/", json={"symbol": "GOOGL", "userid": user_id})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Model for GOOGL trained and saved successfully."

def test_predict_stock():
    # Create a premium user first
    response = client.post("/users/", json={"name": "Eve", "subscription": "premium"})
    assert response.status_code == 200
    user_id = response.json()["userid"]

    # Predict stock prices for a supported stock symbol
    response = client.post("/predict/", json={"symbol": "GOOGL", "userid": user_id})
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "predicted_prices" in data
    assert isinstance(data["predicted_prices"], list)

def test_predict_non_premium_user():
    # Create a basic user
    response = client.post("/users/", json={"name": "Frank", "subscription": "basic"})
    assert response.status_code == 200
    user_id = response.json()["userid"]

    # Try to make a prediction request with a non-premium user
    response = client.post("/predict/", json={"symbol": "GOOGL", "userid": user_id})
    assert response.status_code == 403
    data = response.json()
    assert data == {"detail": "Your membership is not premium. Please upgrade to access this feature."}

def test_invalid_stock_symbol():
    # Create a premium user first
    response = client.post("/users/", json={"name": "Grace", "subscription": "premium"})
    assert response.status_code == 200
    user_id = response.json()["userid"]

    # Attempt to train a model with an unsupported stock symbol
    response = client.post("/train/", json={"symbol": "INVALID", "userid": user_id})
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "Unsupported stock symbol"}
