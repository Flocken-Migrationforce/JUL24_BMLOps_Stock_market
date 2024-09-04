from passlib.context import CryptContext
from auth import load_users_from_csv, update_global_users_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Example of hashing a password
passwords = {
    "admin": "admin",
    "leo": "leodemo",
    "fabian": "fabiandemo",
    "mehdi": "mehdidemo",
    "user": "userdemo"
}

for username, password in passwords.items():
    hashed_password = pwd_context.hash(password)
    print(f"Username: {username}, Hashed Password: {hashed_password}")

# Load users from CSV
loaded_users = load_users_from_csv('database_users.csv')

# Print the loaded users
for username, user in loaded_users.items():
    print(f"Loaded User: {username}, Password: {user.hashed_password}, Subscription: {user.subscription}")

# If you need to update the global users_db
update_global_users_db('database_users.csv')
import sys
sys.exit(0)
# Now you can access users_db globally in auth.py