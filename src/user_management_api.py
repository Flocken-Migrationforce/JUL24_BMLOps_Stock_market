from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import auth

app = FastAPI()

# This should be kept secret and not in the code
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    password: str
    subscription: str

class UserInDB(User):
    hashed_password: str


def get_user(db, username: str):
if username in db:
    user_dict = db[username]
    return UserInDB(**user_dict)

# Function to write a new user to the database_users file
def write_user_to_file(user: User, filename="database_users.txt"):
    try:
        with open(filename, 'a') as file:
            file.write(f"{user.username},{user.password},{user.subscription}\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to file: {str(e)}")

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


@app.post("/users/")
async def create_user(user: User):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)
    fake_users_db[user.username] = {
        "username": user.username,
        "hashed_password": hashed_password,
        "subscription": user.subscription
    }

    # Update the file
    with open(USERS_FILE, 'a') as file:
        file.write(f"{user.username},{user.password},{user.subscription}\n")

    return {"username": user.username, "subscription": user.subscription}

def load_users_from_file(filename):
    users_db = {}
    with open(filename, 'r') as file:
        for line in file:
            username, password, subscription = line.strip().split(',')
            users_db[username] = {
                "username": username,
                "hashed_password": pwd_context.hash(password),
                "subscription": subscription
            }
    return users_db

@app.get("/users/")
async def get_users(current_user: User = Depends(get_current_user)):
    return [{"username": username, "subscription": user["subscription"]}
            for username, user in fake_users_db.items()]


@app.put("/users/{username}")
async def update_user(username: str, user: User, current_user: User = Depends(get_current_user)):
    if username not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")

    fake_users_db[username]["subscription"] = user.subscription

    # Update the file
    with open(USERS_FILE, 'w') as file:
        for u_name, u_data in fake_users_db.items():
            file.write(f"{u_name},{u_data['hashed_password']},{u_data['subscription']}\n")

    return {"username": username, "subscription": user.subscription}


@app.delete("/users/{username}")
async def delete_user(username: str, current_user: User = Depends(get_current_user)):
    if username not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")

    del fake_users_db[username]

    # Update the file
    with open(USERS_FILE, 'w') as file:
        for u_name, u_data in fake_users_db.items():
            file.write(f"{u_name},{u_data['hashed_password']},{u_data['subscription']}\n")

    return {"username": username, "deleted": True}


def load_users_from_file(filename):
    users_db = {}
    with open(filename, 'r') as file:
        for line in file:
            username, password, subscription = line.strip().split(',')
            users_db[username] = {
                "username": username,
                "hashed_password": pwd_context.hash(password),
                "subscription": subscription
            }
    return users_db

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

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

def create_access_token(data: dict, expires_delta: timedelta = None):
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
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user



# Load users from file
USERS_FILE = "users.txt"
fake_users_db = load_users_from_file(USERS_FILE)



@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/users/")
async def create_user(user: User):
    """
    Create a new user and write it to the file database_users.txt.

    Args:
        user (User): The user to create.

    Returns:
        User: The created user.
    """
    # Check if the user already exists
    if user.username in auth.users_db:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Hash the password before storing
    hashed_password = auth.pwd_context.hash(user.password)
    user_data = {
        "username": user.username,
        "hashed_password": hashed_password,
        "subscription": user.subscription
    }

    # Add user to the in-memory database
    auth.users_db[user.username] = user_data

    # Write user to the file
    write_user_to_file(user)

    return {"username": user.username, "subscription": user.subscription}

@app.get("/users/")
async def get_users():
    """
    Retrieve all registered users.

    Returns:
        dict: A dictionary of users keyed by username.
    """
    return [{"username": username, "subscription": user["subscription"]}
            for username, user in auth.users_db.items()]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("user_management:app", host="0.0.0.0", port=8001, reload=True)