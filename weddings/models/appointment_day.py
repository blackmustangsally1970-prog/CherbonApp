from extensions import db

class AppointmentDay(db.Model):
    __tablename__ = "appointment_days"

    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.Date, unique=True, nullable=False)
    is_open = db.Column(db.Boolean, default=True)
