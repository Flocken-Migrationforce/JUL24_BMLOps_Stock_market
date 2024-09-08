import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from httpx import AsyncClient
import pytest
from fastapi import status
from app import app  # Import your FastAPI app

@pytest.mark.asyncio
async def test_admin_login():
    form_data = {
        "username": "admin",
        "password": "admin"
    }
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/token", data=form_data)

        # Check if login was successful and a token was returned
        assert response.status_code == status.HTTP_200_OK
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"

