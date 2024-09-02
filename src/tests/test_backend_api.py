import pytest
from fastapi.testclient import TestClient
from backend_api import app, StockRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_train_model_endpoint():
    valid_request = {"symbol": "AAPL", "userid": 1}
    response = client.post("/train/", json=valid_request)
    assert response.status_code in [200, 400]  # Expect 200 if successful, 400 if unsupported symbol

    invalid_request = {"symbol": "INVALID", "userid": 1}
    response = client.post("/train/", json=invalid_request)
    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported stock symbol"}

@pytest.mark.asyncio
async def test_get_stock_data():
    # Assuming this function needs to be implemented. For now, we can check if the endpoint exists.
    response = client.get("/stock/AAPL")
    assert response.status_code == 200 or response.status_code == 204  # Modify this after implementation
