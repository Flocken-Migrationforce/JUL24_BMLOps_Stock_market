### IAM, User Management (Identification and authorization management)
## 2408282159added


# In-memory database for users
users_db = {}


# Pydantic model for a User
from pydantic import BaseModel
from typing import Optional
<<<<<<< HEAD
=======

>>>>>>> FF
class User(BaseModel):
    """Model representing a user in the application."""
    userid: Optional[int] = None
    name: str
    subscription: str



# Functions
async def get_next_user_id():
    """Helper function to generate the next user ID."""
    return max(users_db.keys(), default=0) + 1