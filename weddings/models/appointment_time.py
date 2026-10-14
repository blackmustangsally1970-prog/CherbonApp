from extensions import db

class AppointmentTime(db.Model):
    __tablename__ = "appointment_times"

    id = db.Column(db.Integer, primary_key=True)
    day_id = db.Column(db.Integer, db.ForeignKey("appointment_days.id"))
    time_label = db.Column(db.String(20), nullable=False)
    is_available = db.Column(db.Boolean, default=True)

    day = db.relationship("AppointmentDay", backref="times")

    @property
    def booking(self):
        from weddings.models.appointment import Appointment
        return Appointment.query.filter_by(
            day=self.day.day,
            time_label=self.time_label
        ).first()
