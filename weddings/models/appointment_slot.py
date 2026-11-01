from extensions import db
from datetime import datetime


class AppointmentSlot(db.Model):
    __tablename__ = "appointment_slots"

    id = db.Column(db.Integer, primary_key=True)

    appointment_date = db.Column(db.Date, nullable=False)

    appointment_time = db.Column(db.String(20), nullable=False)

    is_available = db.Column(
        db.Boolean,
        nullable=False,
        default=True
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )