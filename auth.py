"""Credential check. Users come from the APP_USERS environment variable (or Streamlit secrets).

APP_USERS is a JSON list: [{"username": "admin", "role": "admin", "hash": "<pbkdf2 hash>"}]
Create entries with:  python make_user.py <username> <role>
Passwords are never stored in plain text.
"""
import hashlib
import hmac
import json
import os
import secrets

ITERATIONS = 200_000


def hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITERATIONS).hex()
    return f"{salt}${dk}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


def load_users(raw: str = None):
    raw = raw or os.getenv("APP_USERS", "")
    if not raw:
        return {}
    return {u["username"]: u for u in json.loads(raw)}


def authenticate(username: str, password: str, raw: str = None):
    """Return the user dict on success, else None."""
    user = load_users(raw).get(username)
    if user and verify_password(password, user["hash"]):
        return {"username": username, "role": user.get("role", "agent")}
    return None
