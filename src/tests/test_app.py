import pytest
from fastapi.testclient import TestClient
from app import app, USERS_FILE, load_users_from_csv, write_user_to_file, create_access_token, authenticate_user, get_current_user
from ..auth import verify_password, authenticate_user, get_current_user
from datetime import timedelta

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_train_page():
    response = client.get("/train")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_page():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_create_user():
    user_data = {
        "username": "testuser",
        "password": "testpassword",
        "subscription": "premium"
    }
    response = client.post("/users/", json=user_data)
    assert response.status_code == 200
    assert response.json() == {"username": "testuser", "subscription": "premium"}

def test_get_users():
    response = client.get("/users/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_authenticate_user():
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = client.post("/token", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_update_user():
    update_data = {
        "username": "testuser",
        "password": "newpassword",
        "subscription": "basic"
    }
    response = client.put("/users/testuser", json=update_data)
    assert response.status_code == 200
    assert response.json()["subscription"] == "basic"

def test_delete_user():
    response = client.delete("/users/testuser")
    assert response.status_code == 200
    assert response.json() == {"userid": "testuser", "deleted": True}

def test_predict_stock():
    stock_data = {
        "symbol": "AAPL",
        "userid": "testuser"
    }
    response = client.post("/predict/", json=stock_data)
    assert response.status_code == 403  # Da der Benutzer vermutlich kein Premium-Benutzer ist

def test_train_model():
    stock_data = {
        "symbol": "AAPL",
        "userid": "testuser"
    }
    response = client.post("/train/", json=stock_data)
    assert response.status_code == 400  # Es ist wahrscheinlich, dass der Stock-Symbol nicht unterstützt wird

def test_visualize_stock():
    response = client.get("/visualize/AAPL?days=7")
    assert response.status_code == 500  # Dies könnte variieren, abhängig von der Implementierung

def test_load_users_from_file():
    users_db = load_users_from_file(USERS_FILE)
    assert isinstance(users_db, dict)
    assert "testuser" in users_db

def test_write_user_to_file():
    user = User(username="testuser2", password="testpassword2", subscription="basic")
    write_user_to_file(user, "test_database_users.txt")
    with open("test_database_users.txt", "r") as file:
        lines = file.readlines()
    assert any("testuser2,testpassword2,basic" in line for line in lines)

def test_verify_password():
    password = "testpassword"
    hashed_password = pwd_context.hash(password)
    assert verify_password(password, hashed_password) is True

def test_authenticate_user_function():
    users_db = load_users_from_file(USERS_FILE)
    user = authenticate_user(users_db, "testuser", "testpassword")
    assert user is not False

def test_create_access_token():
    data = {"sub": "testuser"}
    token = create_access_token(data, expires_delta=timedelta(minutes=15))
    assert isinstance(token, str)

def test_get_current_user():
    users_db = load_users_from_file(USERS_FILE)
    token = create_access_token({"sub": "testuser"}, expires_delta=timedelta(minutes=15))
    current_user = get_current_user(token)
    assert current_user.username == "testuser"
