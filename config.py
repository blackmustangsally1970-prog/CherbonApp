import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")
    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Safety check to prevent silent failures
    if not SQLALCHEMY_DATABASE_URI:
        raise RuntimeError("Environment variable SQLALCHEMY_DATABASE_URI is not set.")
