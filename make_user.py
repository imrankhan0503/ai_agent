"""Generate an APP_USERS entry:  python make_user.py <username> <role: admin|agent>"""
import getpass
import json
import sys

from auth import hash_password

if len(sys.argv) < 2:
    sys.exit("usage: python make_user.py <username> [admin|agent]")
username = sys.argv[1]
role = sys.argv[2] if len(sys.argv) > 2 else "agent"
pw = getpass.getpass("Password: ")
print(json.dumps({"username": username, "role": role, "hash": hash_password(pw)}))
print("\nPut one or more of these objects in a JSON list in the APP_USERS variable.")
