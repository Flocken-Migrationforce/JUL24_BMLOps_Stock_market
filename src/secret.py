## ! This is a LOCAL FILE, not to be shared in the Github repo (in productive cases).
## Only for demonstrational purposes, this file was shared with you.
SECRET_KEY = "StockPrediction2024" # for verification of JWT token

'''
Purpose: The SECRET_KEY is used to digitally sign the JWT tokens when they are created and to verify their authenticity when they are received back from the client.
Nature: It's a random string of characters that should be kept secret and known only to the server.
Generation: The SECRET_KEY is typically a randomly generated string. It's not derived or hashed from any other data.
Security: It's crucial for the security of your application. If an attacker gets hold of your SECRET_KEY, they could potentially create valid tokens and impersonate users.
Storage: It should be stored securely, preferably as an environment variable, and not hard-coded in your application code.
Example of generation: You can generate a suitable SECRET_KEY using Python's built-in secrets module or os.urandom():

# to get a string like this, run:
# openssl rand -hex 32

OR:
import secrets
SECRET_KEY = secrets.token_hex(32)

OR:
import os
SECRET_KEY = os.urandom(32).hex()
'''