from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
import csv

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()


# Pydantic model for a User
from pydantic import BaseModel, EmailStr, validator, ValidationError
from typing import Optional, Dict
from enum import Enum


class SubscriptionType(str, Enum):
    BASIC = "basic"
    PREMIUM = "premium"

class User(BaseModel):
    """Model representing a user in the application."""
    userid: Optional[int] = None
    username: str
    full_name: str
    email: EmailStr
    hashed_password: str
    subscription: SubscriptionType


# users_db = {
#     "Leo": {
#         "username": "leo",
#         "full_name": "Leo Loeffler",
#         "hashed_password": pwd_context.hash("leodemo"),
#         "subscription": "premium",
#     },
#     "Fabian": {
#         "username": "fabian",
#         "full_name": "Fabian Flocken",
#         "hashed_password": pwd_context.hash("fabiandemo"),
#         "subscription": "premium",
#     },
#     "Mehdi": {
#         "username": "mehdi",
#         "full_name": "Mir Mehdi Seyedebrahimi",
#         "hashed_password": pwd_context.hash("mehdidemo"),
#         "subscription": "premium",
#     },
#     "User": {
#         "username": "user",
#         "full_name": "Demo User",
#         "hashed_password": pwd_context.hash("userdemo"),
#         "subscription": "basic",
#     },
# }



def load_users_from_csv(filename: str) -> Dict[str, User]:
    users_db = {}
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                user = User(
                    userid=int(row['userid']) if row['userid'] != 'None' else None,
                    username=row['username'],
                    full_name=row['full_name'],
                    email=row['email'],
                    hashed_password=row['hashed_password'],  # Note: This is actually the hashed password
                    subscription=row['subscription']
                )
                users_db[row['username']] = user
            except ValidationError as e:
                print(f"Error validating user data: {e}")
    return users_db


async def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def authenticate_user(username: str, hashed_password: str):
    user = users_db.get(username)
    if not user:
        return False
    if not await verify_password(hashed_password, user['hashed_password']):
        return False
    return user

async def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = await authenticate_user(credentials.username, credentials.password)
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


# get all Users from database
users_db = load_users_from_csv('database_users.csv')
