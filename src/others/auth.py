def authenticate_user(username: str, password: str, users_db: dict) -> bool:
    return username in users_db and users_db[username] == password