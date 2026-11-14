from extensions import db
from datetime import datetime

class AppointmentSlotBlockout(db.Model):
    __tablename__ = "appointment_slot_blockouts"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    appointment_date = db.Column(
        db.Date,
        nullable=False
    )

    appointment_time = db.Column(
        db.String(5),
        nullable=False
    )

    reason = db.Column(
        db.String(200)
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )