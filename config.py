import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ClickSend SMS configuration
    CLICKSEND_USERNAME = os.environ.get("CLICKSEND_USERNAME", "")
    CLICKSEND_API_KEY = os.environ.get("CLICKSEND_API_KEY", "")

    # Dedicated sender numbers
    EQUESTRIAN_SENDER = os.environ.get("EQUESTRIAN_SENDER", "")
    WEDDING_SENDER = os.environ.get("WEDDING_SENDER", "")


    # Build DB URI at class level so Flask can read it
    db_url = os.environ.get("SQLALCHEMY_DATABASE_URI") or \
             os.environ.get("DATABASE_URL") or \
             "sqlite:///mydb.db"

    # Ensure Azure PostgreSQL has SSL enabled
    if db_url.startswith("postgres://") and "sslmode" not in db_url:
        db_url += "?sslmode=require"

    SQLALCHEMY_DATABASE_URI = db_url
