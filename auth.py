"""
Authentication and authorization module for RAG Showdown.
Provides JWT-based authentication with user registration and login.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import logging

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# User models
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    is_active: bool = True
    created_at: datetime

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def init_auth_db(db_path: str = "abstracts.db") -> None:
    """Initialize authentication tables in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create default admin user if no users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    
    if user_count == 0:
        admin_password = get_password_hash("admin123")  # Change this in production
        cursor.execute("""
            INSERT INTO users (username, email, hashed_password, is_active)
            VALUES (?, ?, ?, ?)
        """, ("admin", "admin@example.com", admin_password, True))
        logger.info("Created default admin user (username: admin, password: admin123)")
    
    conn.commit()
    conn.close()

def get_user_by_username(username: str, db_path: str = "abstracts.db") -> Optional[User]:
    """Get user by username from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, hashed_password, is_active, created_at
        FROM users WHERE username = ? AND is_active = TRUE
    """, (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )
    return None

def authenticate_user(username: str, password: str, db_path: str = "abstracts.db") -> Optional[User]:
    """Authenticate user credentials."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, hashed_password, is_active, created_at
        FROM users WHERE username = ? AND is_active = TRUE
    """, (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row and verify_password(password, row["hashed_password"]):
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )
    return None

def create_user(user_data: UserCreate, db_path: str = "abstracts.db") -> User:
    """Create a new user."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (user_data.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    cursor.execute("""
        INSERT INTO users (username, email, hashed_password, is_active)
        VALUES (?, ?, ?, ?)
    """, (user_data.username, user_data.email, hashed_password, True))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return User(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        is_active=True,
        created_at=datetime.now(timezone.utc)
    )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_username(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_user_from_request(request: Request) -> Optional[str]:
    """Extract username from request for rate limiting purposes."""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except:
        return None