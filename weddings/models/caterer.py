from extensions import db

class Caterer(db.Model):
    __tablename__ = 'caterer'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    login_email = db.Column(db.String(120))
    password_hash = db.Column(db.String(200))
