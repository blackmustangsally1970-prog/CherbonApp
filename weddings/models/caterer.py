from extensions import db

class Caterer(db.Model):
    __tablename__ = 'caterer'

    id = db.Column(db.Integer, primary_key=True)

    # Identity
    name = db.Column(db.String(120), nullable=False)

    # Login credentials (already exist)
    login_email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(200))

    # Contact details (new)
    phone = db.Column(db.String(50))
    contact_name = db.Column(db.String(120))
    notes = db.Column(db.Text)

    # Status (new)
    active = db.Column(db.Boolean, default=True)

    # Relationship: weddings assigned to this caterer
    weddings = db.relationship("Wedding", backref="caterer", lazy=True)
