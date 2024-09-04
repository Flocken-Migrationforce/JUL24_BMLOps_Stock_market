from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import csv


# JWT configuration
from secret import SECRET_KEY
# import like e.g.: SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic model for a User
from pydantic import BaseModel, EmailStr, ValidationError
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

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

users_db: Dict[str, User] = {}

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


def load_users_from_csv(filename: str):
    global users_db
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
                    hashed_password=row['hashed_password'],
                    subscription=row['subscription']
                )
                users_db[row['username']] = user
            except ValueError as e:
                print(f"Error validating user data: {e}")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    if username in users_db:
        user_dict = users_db[username].dict()
        return User(**user_dict)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_next_user_id():
    return max([user.userid for user in users_db.values() if user.userid is not None], default=0) + 1

# Load users from CSV file
load_users_from_csv('database_users.csv')