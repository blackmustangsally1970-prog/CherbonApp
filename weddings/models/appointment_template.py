from extensions import db

class AppointmentTemplate(db.Model):
    __tablename__ = "appointment_templates"

    id = db.Column(db.Integer, primary_key=True)

    weekday = db.Column(
        db.Integer,
        nullable=False
    )
    # Monday=0 ... Sunday=6

    appointment_time = db.Column(
        db.String(5),
        nullable=False
    )
    # 07:45

    is_active = db.Column(
        db.Boolean,
        default=True
    )