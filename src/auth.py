from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()


# Pydantic model for a User
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    """Model representing a user in the application."""
    userid: Optional[int] = None
    username: str
    full_name: str
    password: str
    subscription: str


users_db = {
    "Leo": {
        "username": "leo",
        "full_name": "Leo Loeffler",
        "hashed_password": pwd_context.hash("leodemo"),
        "subscription": "premium",
    },
    "Fabian": {
        "username": "fabian",
        "full_name": "Fabian Flocken",
        "hashed_password": pwd_context.hash("fabiandemo"),
        "subscription": "premium",
    },
    "Mehdi": {
        "username": "mehdi",
        "full_name": "Mir Mehdi Seyedebrahimi",
        "hashed_password": pwd_context.hash("mehdidemo"),
        "subscription": "premium",
    },
    "User": {
        "username": "user",
        "full_name": "Demo User",
        "hashed_password": pwd_context.hash("userdemo"),
        "subscription": "basic",
    },
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user:
        return False
    if not verify_password(password, user['hashed_password']):
        return False
    return user

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user

async def get_next_user_id():
    """Helper function to generate the next user ID."""
    return max(users_db.keys(), default=0) + 1
