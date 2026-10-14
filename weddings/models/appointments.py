from extensions import db
from datetime import datetime

class Appointment(db.Model):
    __tablename__ = "appointments"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Client details
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(40), nullable=False)

    # Appointment details
    day = db.Column(db.Date, nullable=False)
    time_label = db.Column(db.String(40), nullable=False)

    # Optional client-provided info
    wedding_date = db.Column(db.Date)
    guest_numbers = db.Column(db.Integer)

    # Optional notes from the client
    notes = db.Column(db.Text)

    # Workflow status
    # booked → viewed → quoted → closed
    status = db.Column(db.String(40), default="booked")
