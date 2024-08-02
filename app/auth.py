# User authentication logic (via FastAPI) : RSA-JWT (JSON Web Token)
# Version 0.2
# Fabian
# 2408011207


'''
Authentication roadmap:
1. Initialisation: JWT-related imports
2. RSA key generation for token signing
3. Token creation functions
4. Token verification
5. Access and refresh token mechanisms
6. Token-based authentication flow

User registration (POST /users/)
User login and token generation (POST /token)
Retrieving current user information (GET /users/me/)
Retrieving user information by username (GET /users/{username})


Algorithm for signing:
The code uses HS256 (HMAC with SHA-256), which is not one of the recommended algorithms.
(It's not insecure, but EdDSA, ES256, or RS256 would be preferable.)

Token live for 5 hours:
The code sets ACCESS_TOKEN_EXPIRE_MINUTES = 300
Token refresh mechanism:
The code doesn't implement a token refresh mechanism, which would be an important addition for better security and user experience.
Secure token storage:
The code doesn't specify how tokens should be stored client-side. Ideally, this should be done in HTTP-only cookies.
HTTPS transmission:
The code doesn't enforce HTTPS, which would be crucial in a production environment.
'''


# app/auth.py
from fastapi import FastAPI, Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# Generate RSA key pair
private_key = rsa.generate_private_key(
	public_exponent=65537,
	key_size=2048
)
public_key = private_key.public_key()

# Serialize keys to PEM format
PRIVATE_KEY = private_key.private_bytes(
	encoding=serialization.Encoding.PEM,
	format=serialization.PrivateFormat.PKCS8,
	encryption_algorithm=serialization.NoEncryption()
)
PUBLIC_KEY = public_key.public_bytes(
	encoding=serialization.Encoding.PEM,
	format=serialization.PublicFormat.SubjectPublicKeyInfo
)

ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300
REFRESH_TOKEN_EXPIRE_DAYS = 14

fake_users_db = {
# users_db = {
    "admin": "admin",
    "fabian": "fabianpass",
    "mehdi": "mehdipass",
    "leo": "leopass",
    "florian": "florianpass",
	"user": "user"
}

class Token(BaseModel):
	access_token: str
	token_type: str


class TokenData(BaseModel):
	username: Optional[str] = None


class User(BaseModel):
	username: str
	email: Optional[str] = None
	full_name: Optional[str] = None
	disabled: Optional[bool] = None


class UserInDB(User):
	hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

auth_app = FastAPI()

def verify_password(plain_password, hashed_password):
	return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
	return pwd_context.hash(password)


def get_user(db, username: str):
	if username in db:
		user_dict = db[username]
		return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
	user = get_user(fake_db, username)
	if not user:
		return False
	if not verify_password(password, user.hashed_password):
		return False
	return user


def create_token(data: dict, expires_delta: timedelta):
	to_encode = data.copy()
	expire = datetime.utcnow() + expires_delta
	to_encode.update({"exp": expire})
	encoded_jwt = jwt.encode(to_encode, PRIVATE_KEY, algorithm=ALGORITHM)
	return encoded_jwt


def create_access_token(data: dict):
	return create_token(data, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))


def create_refresh_token(data: dict):
	return create_token(data, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))


async def get_current_user(token: str = Depends(oauth2_scheme)):
	credentials_exception = HTTPException(
		status_code=status.HTTP_401_UNAUTHORIZED,
		detail="Could not validate credentials",
		headers={"WWW-Authenticate": "Bearer"},
	)
	try:
		payload = jwt.decode(token, PUBLIC_KEY, algorithms=[ALGORITHM])
		username: str = payload.get("sub")
		if username is None:
			raise credentials_exception
		token_data = TokenData(username=username)
	except JWTError:
		raise credentials_exception
	user = get_user(fake_users_db, username=token_data.username)
	if user is None:
		raise credentials_exception
	return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
	if current_user.disabled:
		raise HTTPException(status_code=400, detail="Inactive user")
	return current_user


@auth_app.post("/token")
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
	user = authenticate_user(fake_users_db, form_data.username, form_data.password)
	if not user:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Incorrect username or password",
			headers={"WWW-Authenticate": "Bearer"},
		)
	access_token = create_access_token(data={"sub": user.username})
	refresh_token = create_refresh_token(data={"sub": user.username})

	response.set_cookie(
		key="refresh_token",
		value=refresh_token,
		httponly=True,
		secure=True,
		samesite="strict",
		max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
	)

	return {"access_token": access_token, "token_type": "bearer"}


@auth_app.post("/refresh")
async def refresh_token(request: Request, response: Response):
	refresh_token = request.cookies.get("refresh_token")
	if not refresh_token:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Refresh token missing",
		)
	try:
		payload = jwt.decode(refresh_token, PUBLIC_KEY, algorithms=[ALGORITHM])
		username: str = payload.get("sub")
		if username is None:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail="Invalid refresh token",
			)
		access_token = create_access_token(data={"sub": username})
		return {"access_token": access_token, "token_type": "bearer"}
	except JWTError:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid refresh token",
		)


@auth_app.post("/users/", response_model=User)
async def create_user(user: UserInDB):
	if user.username in fake_users_db:
		raise HTTPException(status_code=400, detail="Username already registered")
	hashed_password = get_password_hash(user.hashed_password)
	user_dict = user.dict()
	user_dict["hashed_password"] = hashed_password
	fake_users_db[user.username] = user_dict
	return user_dict


@auth_app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
	return current_user


@auth_app.get("/users/{username}", response_model=User)
async def read_user(username: str, current_user: User = Depends(get_current_active_user)):
	if username != current_user.username:
		raise HTTPException(status_code=400, detail="Can only access own user information")
	return current_user


# Middleware to enforce HTTPS
@auth_app.middleware("http")
async def enforce_https(request: Request, call_next):
	if request.url.scheme != "https":
		return Response("HTTPS required", status_code=403)
	return await call_next(request)