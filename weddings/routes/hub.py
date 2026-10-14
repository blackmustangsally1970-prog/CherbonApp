from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from flask_login import login_required, current_user
from datetime import datetime
from extensions import db
from auth.permissions import admin_required, is_coordinator, is_caterer

from weddings.models.appointments import Appointment
from weddings.models.appointment_day import AppointmentDay
from weddings.models.appointment_time import AppointmentTime
from weddings.models.appointment_blockout import AppointmentBlockout

from weddings.models.wedding import Wedding
from weddings.models.staff_assignment import WeddingStaffAssignment
from weddings.models.job_list_instance import JobListInstance
from weddings.models.job_list_master import JobListMaster
from weddings.models.wedding_timeline import WeddingTimeline
from weddings.models.wedding_detail_library import WeddingDetailLibrary
from weddings.models.caterer import Caterer

from weddings.emailer import send_email

hub_bp = Blueprint('hub', __name__, url_prefix='/weddings')


# ---------------------------------------------------------
# MASTER PAGE – ALL WEDDINGS
# ---------------------------------------------------------

@hub_bp.route('/')
@login_required
def weddings_master():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    weddings = Wedding.query.filter(
        Wedding.booking_status.notin_(["archived", "cancelled", "postponed"])
    ).order_by(
        Wedding.wedding_date.asc(),
        Wedding.start_time.asc()
    ).all()

    return render_template('weddings/master.html', weddings=weddings)

@hub_bp.route('/<int:wedding_id>/hub')
@login_required
def wedding_hub(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)

    # Optional: restrict access
    if current_user.role not in ['admin', 'management', 'coordinator', 'manager']:
        abort(403)

    return render_template('weddings/hub.html', wedding=w)


@hub_bp.route('/postponed')
@login_required
def postponed_weddings():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    postponed = Wedding.query.filter(
        Wedding.booking_status == "postponed"
    ).order_by(
        Wedding.wedding_date.asc()
    ).all()

    return render_template('weddings/postponed.html', weddings=postponed)


@hub_bp.route('/<int:wedding_id>/postpone', methods=['POST'])
@login_required
def postpone_wedding(wedding_id):
    w = Wedding.query.get_or_404(wedding_id)

    if current_user.role not in ['admin', 'management']:
        abort(403)

    w.booking_status = "postponed"
    w.postponed_at = datetime.now()

    db.session.commit()

    flash("Wedding postponed. It will appear in the Postponed Weddings list.", "warning")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))




@hub_bp.route('/archived')
@login_required
def archived_weddings():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    archived = Wedding.query.filter(
        Wedding.booking_status.in_(["archived", "cancelled"])
    ).order_by(
        Wedding.wedding_date.desc()
    ).all()

    return render_template('weddings/archived.html', weddings=archived)


# ---------------------------------------------------------
# CREATE HOLD (ADMIN ONLY)
# ---------------------------------------------------------
@hub_bp.route('/create_hold', methods=['GET', 'POST'])
@admin_required
def create_hold():

    if request.method == 'POST':
        date = request.form.get('hold_date')
        name = request.form.get('hold_name')
        notes = request.form.get('hold_notes')

        # Convert date
        from datetime import datetime
        hold_date = datetime.strptime(date, "%Y-%m-%d").date()

        # Check if already booked or held
        existing = Wedding.query.filter_by(wedding_date=hold_date).first()
        if existing:
            flash("This date already has a booking or hold.", "danger")
            return redirect(url_for('hub.create_hold'))

        # Create hold entry
        w = Wedding(
            wedding_date=hold_date,
            is_hold=True,
            hold_name=name,
            hold_notes=notes,
            hold_created_at=datetime.now(),
            booking_status="hold"
        )

        db.session.add(w)
        db.session.commit()

        flash("Hold created.", "success")
        return redirect(url_for('hub.weddings_master'))

    return render_template('weddings/create_hold.html')

@hub_bp.route('/<int:wedding_id>/archive', methods=['POST'])
@login_required
def archive_wedding(wedding_id):
    w = Wedding.query.get_or_404(wedding_id)

    if current_user.role not in ['admin', 'management']:
        abort(403)

    w.booking_status = "archived"
    w.archived_at = datetime.now()

    db.session.commit()

    flash("Wedding archived successfully.", "info")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


@hub_bp.route('/<int:wedding_id>/cancel', methods=['POST'])
@login_required
def cancel_wedding(wedding_id):
    w = Wedding.query.get_or_404(wedding_id)

    if current_user.role not in ['admin', 'management']:
        abort(403)

    w.booking_status = "cancelled"
    w.cancelled_at = datetime.now()

    db.session.commit()

    flash("Wedding cancelled successfully.", "warning")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# REMOVE HOLD (ADMIN ONLY)
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/remove_hold', methods=['POST'])
@admin_required
def remove_hold(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)

    if not w.is_hold:
        flash("This is not a hold.", "danger")
        return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

    # Convert hold back to pending admin booking
    w.is_hold = False
    w.booking_status = "pending_admin"
    w.hold_name = None
    w.hold_notes = None
    w.hold_created_at = None
    w.hold_expires_at = None

    db.session.commit()

    flash("Hold removed.", "success")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

# ---------------------------------------------------------
# INCOMING T&C QUEUE (ADMIN ONLY)
# ---------------------------------------------------------
@hub_bp.route('/incoming')
@admin_required
def incoming_queue():

    incoming = Wedding.query.filter(
        Wedding.booking_status == "pending_admin",
        Wedding.is_hold == False
    ).order_by(Wedding.submission_date.desc()).all()

    return render_template(
        'weddings/incoming.html',
        incoming=incoming
    )

# ---------------------------------------------------------
# APPROVE BOOKING
# ---------------------------------------------------------
@hub_bp.route('/incoming/<int:wedding_id>/approve', methods=['POST'])
@admin_required
def approve_booking(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)

    # Double-booking protection
    conflict = Wedding.query.filter(
        Wedding.wedding_date == w.wedding_date,
        Wedding.id != w.id,
        (Wedding.booking_status != "cancelled") | (Wedding.is_hold == True)
    ).first()

    if conflict:
        flash("Cannot approve — date already booked or on hold.", "danger")
        return redirect(url_for('hub.incoming_queue'))

    from datetime import datetime
    w.booking_status = "approved"
    w.admin_approved_at = datetime.now()

    db.session.commit()

    flash("Booking approved.", "success")
    return redirect(url_for('hub.weddings_master'))

# ---------------------------------------------------------
# REJECT BOOKING
# ---------------------------------------------------------
@hub_bp.route('/incoming/<int:wedding_id>/reject', methods=['POST'])
@admin_required
def reject_booking(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.booking_status = "cancelled"

    db.session.commit()

    flash("Booking rejected.", "info")
    return redirect(url_for('hub.incoming_queue'))




# ---------------------------------------------------------
# NOTIFY DECORATOR
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/notify/decorator')
@admin_required
def notify_decorator(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.decorator_notified_at = datetime.now()

    if w.booking_status == "approved":
        w.booking_status = "vendor_notified"

    db.session.commit()

    flash("Decorator notified.", "success")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# NOTIFY COORDINATOR
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/notify/coordinator')
@admin_required
def notify_coordinator(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.coordinator_notified_at = datetime.now()

    if w.booking_status == "approved":
        w.booking_status = "vendor_notified"

    db.session.commit()

    flash("Coordinator notified.", "success")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# NOTIFY STAFF
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/notify/staff')
@admin_required
def notify_staff(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.staff_notified_at = datetime.now()

    if w.booking_status == "approved":
        w.booking_status = "vendor_notified"

    db.session.commit()

    flash("Staff notified.", "success")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))



@hub_bp.route('/<int:wedding_id>/confirm/decorator')
def confirm_decorator(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.decorator_confirmed_at = datetime.now()

    if all([
        w.caterer_confirmed_at,
        w.decorator_confirmed_at,
        w.coordinator_confirmed_at,
        w.staff_confirmed_at
    ]):
        w.booking_status = "vendor_confirmed"

    db.session.commit()

    return render_template('vendor/confirm.html', wedding=w)


# ---------------------------------------------------------
# CONFIRM: CATERER
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/confirm/caterer')
def confirm_caterer(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.caterer_confirmed_at = datetime.now()

    if all([
        w.caterer_confirmed_at,
        w.decorator_confirmed_at,
        w.coordinator_confirmed_at,
        w.staff_confirmed_at
    ]):
        w.booking_status = "vendor_confirmed"

    db.session.commit()

    return render_template('vendor/confirm.html', wedding=w)

# ---------------------------------------------------------
# CONFIRM: COORDINATOR
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/confirm/coordinator')
def confirm_coordinator(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.coordinator_confirmed_at = datetime.now()

    if all([
        w.caterer_confirmed_at,
        w.decorator_confirmed_at,
        w.coordinator_confirmed_at,
        w.staff_confirmed_at
    ]):
        w.booking_status = "vendor_confirmed"

    db.session.commit()

    return render_template('vendor/confirm.html', wedding=w)

# ---------------------------------------------------------
# CONFIRM: STAFF
# ---------------------------------------------------------

@hub_bp.route('/<int:wedding_id>/confirm/staff')
def confirm_staff(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.staff_confirmed_at = datetime.now()

    if all([
        w.caterer_confirmed_at,
        w.decorator_confirmed_at,
        w.coordinator_confirmed_at,
        w.staff_confirmed_at
    ]):
        w.booking_status = "vendor_confirmed"

    db.session.commit()

    return render_template('vendor/confirm.html', wedding=w)


# ---------------------------------------------------------
# DECLINE: ALL
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/decline/caterer')
def decline_caterer(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.caterer_confirmed_at = None
    w.booking_status = "vendor_declined"

    db.session.commit()

    return render_template('vendor/decline.html', wedding=w)

@hub_bp.route('/<int:wedding_id>/decline/decorator')
def decline_decorator(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.decorator_confirmed_at = None
    w.booking_status = "vendor_declined"

    db.session.commit()

    return render_template('vendor/decline.html', wedding=w)

@hub_bp.route('/<int:wedding_id>/decline/coordinator')
def decline_coordinator(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.coordinator_confirmed_at = None
    w.booking_status = "vendor_declined"

    db.session.commit()

    return render_template('vendor/decline.html', wedding=w)

@hub_bp.route('/<int:wedding_id>/decline/staff')
def decline_staff(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.staff_confirmed_at = None
    w.booking_status = "vendor_declined"

    db.session.commit()

    return render_template('vendor/decline.html', wedding=w)



# ---------------------------------------------------------
# NOTIFY CATERER
# ---------------------------------------------------------

@hub_bp.route('/<int:wedding_id>/notify/caterer')
@admin_required
def notify_caterer(wedding_id):

    w = Wedding.query.get_or_404(wedding_id)
    w.caterer_notified_at = datetime.now()

    if w.booking_status == "approved":
        w.booking_status = "vendor_notified"

    db.session.commit()

    subject = f"Cherbon Waters – Wedding Notification ({w.bride_name} & {w.groom_name})"
    body = f"""
Hello,

You have been assigned as the caterer for the following wedding at Cherbon Waters:

Bride & Groom: {w.bride_name} & {w.groom_name}
Date: {w.wedding_date.strftime('%d/%m/%Y')}
Start Time: {w.start_time}
Estimated Guests: {w.est_guests}
Event Type: {w.event_type}
Other Information: {w.other_information or 'None provided'}

Please confirm your availability:

Confirm: https://cherbonwaters.com.au/weddings/{w.id}/confirm/caterer
Decline: https://cherbonwaters.com.au/weddings/{w.id}/decline/caterer

Thank you,
Cherbon Waters Management
"""

    send_email(w.caterer.email, subject, body)

    flash("Caterer notified.", "success")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# VIEWING APPOINTMENTS (PUBLIC + ADMIN)
# ---------------------------------------------------------

@hub_bp.route('/appointments', methods=['GET', 'POST'])
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

        # Optional: SMS confirmation (ClickSend)
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


# ---------------------------------------------------------
# ADMIN: LIST ALL APPOINTMENTS
# ---------------------------------------------------------

@hub_bp.route('/admin/appointments')
@login_required
def appointments_admin():
    if current_user.role not in ['admin', 'management']:
        abort(403)

    appts = Appointment.query.order_by(
        Appointment.day.asc(),
        Appointment.time_label.asc()
    ).all()

    return render_template('weddings/appointments_admin.html', appointments=appts)


# ---------------------------------------------------------
# ADMIN: MARK VIEWED
# ---------------------------------------------------------

@hub_bp.route('/admin/appointments/<int:appt_id>/viewed', methods=['POST'])
@login_required
def appointment_mark_viewed(appt_id):
    if current_user.role not in ['admin', 'management']:
        abort(403)

    appt = Appointment.query.get_or_404(appt_id)
    appt.status = 'viewed'
    db.session.commit()

    flash("Appointment marked as viewed.", "info")
    return redirect(url_for('hub.appointments_admin'))


# ---------------------------------------------------------
# ADMIN: SEND FOLLOW-UP PDF
# ---------------------------------------------------------

@hub_bp.route('/admin/appointments/<int:appt_id>/send_followup', methods=['POST'])
@login_required
def appointment_send_followup(appt_id):
    if current_user.role not in ['admin', 'management']:
        abort(403)

    appt = Appointment.query.get_or_404(appt_id)

    # Generate PDF using your existing WeasyPrint wrapper
    pdf_bytes = render_pdf(
        'weddings/appointments_followup_pdf.html',
        appointment=appt
    )

    # Email it using your existing wedding emailer
    send_email(
        appt.email,
        "Cherbon Waters – Viewing Follow‑Up",
        "Thank you for visiting Cherbon Waters. Please find your follow‑up information attached.",
        pdf_bytes
    )

    appt.status = 'quoted'
    db.session.commit()

    flash("Follow‑up PDF sent.", "success")
    return redirect(url_for('hub.appointments_admin'))


@hub_bp.route('/admin/appointments/schedule', methods=['GET', 'POST'])
@login_required
def appointments_schedule():
    if current_user.role not in ['admin', 'management']:
        abort(403)

    from datetime import date, timedelta

    # Range: next 30 days
    today = date.today()
    end = today + timedelta(days=30)

    days = AppointmentDay.query.filter(
        AppointmentDay.day >= today,
        AppointmentDay.day <= end
    ).order_by(AppointmentDay.day.asc()).all()

    # Ensure days exist for range
    existing_dates = {d.day for d in days}
    for i in range(31):
        d = today + timedelta(days=i)
        if d not in existing_dates:
            new_day = AppointmentDay(day=d, is_open=False)
            db.session.add(new_day)
            days.append(new_day)
    db.session.commit()

    # Sort again after inserts
    days = sorted(days, key=lambda d: d.day)

    # Generate 45-min slots if none exist for open days
    default_slots = [
        "9:00 AM", "9:45 AM", "10:30 AM",
        "11:15 AM", "12:00 PM", "12:45 PM",
        "1:30 PM", "2:15 PM", "3:00 PM", "3:45 PM"
    ]

    for d in days:
        if d.is_open and not d.times:
            for label in default_slots:
                db.session.add(AppointmentTime(day=d, time_label=label, is_available=True))
    db.session.commit()

    # Blockouts for display
    blockouts = AppointmentBlockout.query.order_by(
        AppointmentBlockout.start_date.asc()
    ).all()

    return render_template(
        'weddings/appointments_schedule.html',
        days=days,
        blockouts=blockouts
    )

@hub_bp.route('/admin/appointments/day/<int:day_id>/toggle', methods=['POST'])
@login_required
def appointments_toggle_day(day_id):
    if current_user.role not in ['admin', 'management']:
        abort(403)

    d = AppointmentDay.query.get_or_404(day_id)
    d.is_open = not d.is_open
    db.session.commit()

    flash("Day availability updated.", "info")
    return redirect(url_for('hub.appointments_schedule'))


@hub_bp.route('/admin/appointments/time/<int:time_id>/toggle', methods=['POST'])
@login_required
def appointments_toggle_time(time_id):
    if current_user.role not in ['admin', 'management']:
        abort(403)

    t = AppointmentTime.query.get_or_404(time_id)
    t.is_available = not t.is_available
    db.session.commit()

    flash("Time slot updated.", "info")
    return redirect(url_for('hub.appointments_schedule'))


@hub_bp.route('/admin/appointments/blockout', methods=['POST'])
@login_required
def appointments_blockout():
    if current_user.role not in ['admin', 'management']:
        abort(403)

    from datetime import datetime

    start = request.form['start_date']
    end = request.form['end_date']
    reason = request.form.get('reason', '')

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    b = AppointmentBlockout(
        start_date=start_date,
        end_date=end_date,
        reason=reason
    )
    db.session.add(b)

    # Mark days in range as closed
    days = AppointmentDay.query.filter(
        AppointmentDay.day >= start_date,
        AppointmentDay.day <= end_date
    ).all()
    for d in days:
        d.is_open = False

    db.session.commit()

    flash("Blockout range added.", "warning")
    return redirect(url_for('hub.appointments_schedule'))
