from flask import Blueprint, render_template, request
from weddings.models.appointment_day import AppointmentDay
from weddings.models.appointment_time import AppointmentTime
from weddings.models.appointments import Appointment

public_bp = Blueprint("public", __name__)

@public_bp.route("/appointments", methods=["GET", "POST"])
def appointments_public():
    # Load days + times
    days = AppointmentDay.query.order_by(AppointmentDay.day.asc()).all()

    selected_day = request.args.get("day")
    selected_time = request.args.get("time")

    times = []
    if selected_day:
        times = AppointmentTime.query.filter_by(day=selected_day).order_by(AppointmentTime.time_label.asc()).all()

    return render_template(
        "weddings/appointments_public.html",
        days=days,
        times=times,
        selected_day=selected_day,
        selected_time=selected_time
    )
