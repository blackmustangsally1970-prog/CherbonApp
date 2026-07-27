from flask import session
from functools import wraps

# -----------------------------
# ROLE CHECKERS
# -----------------------------
def is_admin():
    return session.get('role') in ['admin', 'management']

def is_management():
    return session.get('role') == 'management'

def is_coordinator():
    return session.get('role') == 'coordinator'

def is_caterer():
    return session.get('role') == 'caterer'


# -----------------------------
# DECORATORS
# -----------------------------
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_admin():
            return "Unauthorized", 403
        return f(*args, **kwargs)
    return wrapper


def coordinator_blocked(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if is_coordinator():
            return "Unauthorized", 403
        return f(*args, **kwargs)
    return wrapper


def caterer_only(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_caterer():
            return "Unauthorized", 403
        return f(*args, **kwargs)
    return wrapper
