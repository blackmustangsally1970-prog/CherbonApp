from flask import Blueprint, render_template, request

# Correct imports — your models live in weddings/models
from weddings.models.appointments import Appointment
from extensions import db

# Your SMS module *is* inside CherbonApp, so this import is correct
from CherbonApp.sms import send_sms

public_bp = Blueprint('public', __name__)

@public_bp.route('/appointments', methods=['GET', 'POST'])
def appointments_public():
    if request.method == 'POST':
        appt = Appointment(
            name=request.form['name'],
            email=request.form['email'],
            phone=request.form['phone'],
            day=request.form['day'],
            time_label=request.form['time'],
            notes=request.form.get('notes', '')
        )

        db.session.add(appt)
        db.session.commit()

        try:
            send_sms(
                appt.phone,
                f"Hi {appt.name}, your Cherbon Waters appointment is booked for "
                f"{appt.day} at {appt.time_label}."
            )
        except Exception as e:
            print("SMS error:", e)

        return render_template('weddings/appointments_thankyou.html')

    return render_template('weddings/appointments_public.html')
