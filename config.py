import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")

    db_url = os.environ.get("SQLALCHEMY_DATABASE_URI")
    if db_url:
        print("Using SQLALCHEMY_DATABASE_URI")
    else:
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            print("Using DATABASE_URL")

    if not db_url:
        print("WARNING: No DB URL found, falling back to SQLite")
        db_url = "sqlite:///mydb.db"

    SQLALCHEMY_DATABASE_URI = db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False
