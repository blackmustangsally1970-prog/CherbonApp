import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")

    # Try SQLALCHEMY_DATABASE_URI first (your preferred variable)
    db_url = os.environ.get("SQLALCHEMY_DATABASE_URI")

    # If Azure injects DATABASE_URL, use it
    if not db_url:
        db_url = os.environ.get("DATABASE_URL")

    # Final fallback to SQLite
    if not db_url:
        db_url = "sqlite:///mydb.db"

    SQLALCHEMY_DATABASE_URI = db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False
