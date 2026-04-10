import json
import hashlib
import os
import datetime

# User database file
USER_DB_FILE = "users.json"


def hash_password(password):
    """Hash password using PBKDF2-HMAC-SHA256 with a random salt.
    Returns 'salt_hex$key_hex'.
    """
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
    return salt.hex() + '$' + key.hex()


def verify_password(password, stored):
    """Verify a password against a stored hash.
    Supports both the new salted format and the legacy unsalted SHA-256 format.
    """
    if '$' in stored:
        salt_hex, key_hex = stored.split('$', 1)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
        return key.hex() == key_hex
    # Legacy: plain SHA-256 — still works for old accounts
    return hashlib.sha256(password.encode()).hexdigest() == stored


def load_users():
    """Load users from JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users):
    """Save users to JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)


def signup(username, password):
    """Create new user account"""
    users = load_users()

    # Check if username already exists
    if username in users:
        return False, "Username already exists!"

    # Validate username and password
    if len(username) < 3:
        return False, "Username must be at least 3 characters!"

    if len(password) < 6:
        return False, "Password must be at least 6 characters!"

    # Create new user
    users[username] = {
        'password': hash_password(password),
        'created_at': str(datetime.datetime.now()),
        'chat_history': []
    }

    save_users(users)
    return True, "Account created successfully!"


def login(username, password):
    """Verify user credentials"""
    users = load_users()

    # Check if user exists
    if username not in users:
        return False, "Username not found!"

    # Check password
    if not verify_password(password, users[username]['password']):
        return False, "Incorrect password!"

    return True, "Login successful!"


def user_exists(username):
    """Check if username exists"""
    users = load_users()
    return username in users


def save_user_chat(username, chat_history):
    """Save chat history for a user"""
    users = load_users()

    if username in users:
        # Add timestamp to chat
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create chat session
        chat_session = {
            'timestamp': timestamp,
            'messages': chat_history
        }

        # Initialize chat_sessions list if doesn't exist
        if 'chat_sessions' not in users[username]:
            users[username]['chat_sessions'] = []

        # Add new session
        users[username]['chat_sessions'].append(chat_session)

        # Keep only last 20 sessions
        users[username]['chat_sessions'] = users[username]['chat_sessions'][-20:]

        save_users(users)
        return True
    return False


def update_user_chat(username, chat_history):
    """Update the most recent chat session instead of creating a new one"""
    users = load_users()

    if username in users:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        chat_session = {
            'timestamp': timestamp,
            'messages': chat_history
        }

        if 'chat_sessions' not in users[username]:
            users[username]['chat_sessions'] = []

        # Update last session instead of creating new one
        if len(users[username]['chat_sessions']) > 0:
            users[username]['chat_sessions'][-1] = chat_session
        else:
            # First chat, create new
            users[username]['chat_sessions'].append(chat_session)

        save_users(users)
        return True
    return False

def get_user_chats(username):
    """Get all saved chats for a user"""
    users = load_users()

    if username in users and 'chat_sessions' in users[username]:
        return users[username]['chat_sessions']
    return []


def delete_user_chat(username, chat_index):
    """Delete a specific chat session"""
    users = load_users()

    if username in users and 'chat_sessions' in users[username]:
        if 0 <= chat_index < len(users[username]['chat_sessions']):
            del users[username]['chat_sessions'][chat_index]
            save_users(users)
            return True
    return False