class Account(db.Model):
    __tablename__ = "accounts"

    id = db.Column(db.Integer, primary_key=True)

    # Identity
    full_name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(30))
    contact_name = db.Column(db.String(120))
    notes = db.Column(db.Text)
    active = db.Column(db.Boolean, default=True)

    # Login (email + password)
    username = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(200))

    # PIN Login (staff/teacher/strapper)
    setup_code = db.Column(db.String(20), unique=True)
    pin_hash = db.Column(db.String(200))
    pin_failures = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)

    # Role System
    primary_role = db.Column(db.String(50), nullable=False)

    is_admin = db.Column(db.Boolean, default=False)
    is_management = db.Column(db.Boolean, default=False)
    is_coordinator = db.Column(db.Boolean, default=False)
    is_staff = db.Column(db.Boolean, default=False)
    is_caterer = db.Column(db.Boolean, default=False)
    is_teacher = db.Column(db.Boolean, default=False)
    is_strapper = db.Column(db.Boolean, default=False)

    # Flask-Login
    def get_id(self):
        return str(self.id)
