
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
#from app import app


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.auth import verify_password, authenticate_user, get_current_user, get_next_user_id, users_db
from src.app import app

# from auth import verify_password, authenticate_user, get_current_user, get_next_user_id, users_db
# from ..auth import verify_password, authenticate_user, get_current_user, get_next_user_id, users_db
# from ..app import app
from fastapi.security import HTTPBasicCredentials
from passlib.context import CryptContext

client = TestClient(app)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@pytest.mark.asyncio
async def test_verify_password():
    plain_password = "testpassword"
    hashed_password = pwd_context.hash(plain_password)
    assert await verify_password(plain_password, hashed_password) == True

@pytest.mark.asyncio
async def test_authenticate_user():
    # Test with correct credentials
    user = await authenticate_user("leo", "leodemo")
    assert user is not False
    assert user["username"] == "leo"

    # Test with incorrect username
    user = await authenticate_user("unknown", "leodemo")
    assert user is False

    # Test with incorrect password
    user = await authenticate_user("leo", "wrongpassword")
    assert user is False

@pytest.mark.asyncio
async def test_get_current_user():
    # Test with correct credentials
    credentials = HTTPBasicCredentials(username="leo", password="leodemo")
    user = await get_current_user(credentials)
    assert user["username"] == "leo"

    # Test with incorrect credentials
    credentials = HTTPBasicCredentials(username="leo", password="wrongpassword")
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(credentials)
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert excinfo.value.detail == "Incorrect username or password"

@pytest.mark.asyncio
async def test_get_next_user_id():
    next_id = await get_next_user_id()
    assert isinstance(next_id, int)
    assert next_id > 0
