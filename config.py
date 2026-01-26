import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    def __init__(self):
        # Prefer explicit SQLALCHEMY_DATABASE_URI
        db_url = os.environ.get("SQLALCHEMY_DATABASE_URI")

        # Fallback to DATABASE_URL (Azure convention)
        if not db_url:
            db_url = os.environ.get("DATABASE_URL")

        # If still nothing, fallback to SQLite
        if not db_url:
            db_url = "sqlite:///mydb.db"

        # Ensure Azure PostgreSQL has SSL enabled
        if db_url.startswith("postgres://") and "sslmode" not in db_url:
            db_url += "?sslmode=require"

        self.SQLALCHEMY_DATABASE_URI = db_url
