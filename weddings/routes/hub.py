from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from flask_login import login_required, current_user
from datetime import datetime, date, timedelta
from extensions import db
from auth.permissions import admin_required, is_coordinator, is_caterer

from utils.sms import send_sms
from weddings.models.appointments import Appointment
from weddings.models.appointment_day import AppointmentDay
from weddings.models.appointment_time import AppointmentTime
from weddings.models.appointment_blockout import AppointmentBlockout
from weddings.models.appointment_slot import AppointmentSlot
from weddings.models.appointment_template import AppointmentTemplate
from weddings.models.appointment_slot_blockouts import AppointmentSlotBlockout



from weddings.models.wedding import Wedding
from weddings.models.staff_assignment import WeddingStaffAssignment
from weddings.models.job_list_instance import JobListInstance
from weddings.models.job_list_master import JobListMaster
from weddings.models.timeline import WeddingTimeline
from weddings.models.detail_template import WeddingDetailLibrary
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


# ---------------------------------------------------------
# APPOINTMENTS DASHBOARD
# ---------------------------------------------------------

@hub_bp.route('/admin/appointments-dashboard')
@login_required
def appointments_dashboard():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    bookings_count = Appointment.query.filter(
        Appointment.status == 'booked'
    ).count()

    pipeline_count = Appointment.query.filter(
        Appointment.status == 'quoted'
    ).count()

    return render_template(
        'weddings/appointments_dashboard.html',
        bookings_count=bookings_count,
        pipeline_count=pipeline_count
    )


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
@login_required
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
# ADMIN: SLOT DASHBOARD
# ---------------------------------------------------------

@hub_bp.route('/admin/slots')
@login_required
def appointment_slots():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    slots = AppointmentSlot.query.filter(
        AppointmentSlot.appointment_date >= date.today()
    ).order_by(
        AppointmentSlot.appointment_date.asc(),
        AppointmentSlot.appointment_time.asc()
    ).all()

    return render_template(
        'weddings/appointment_slots.html',
        slots=slots
    )

# ---------------------------------------------------------
# REMOVE HOLD (ADMIN ONLY)
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/remove_hold', methods=['POST'])
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
# ADMIN: LIST ALL APPOINTMENTS
# ---------------------------------------------------------

@hub_bp.route('/admin/appointments')
@login_required
def appointments_admin():
    if current_user.role not in ['admin', 'management']:
        abort(403)

    appts = Appointment.query.filter(
        Appointment.status.in_([
            'booked',
            'viewed'
        ])
    ).order_by(
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

@hub_bp.route('/admin/appointments/<int:appt_id>/send_followup',
              methods=['POST'])
@login_required
def appointment_send_followup(appt_id):

    if current_user.role not in ['admin', 'management']:
        abort(403)

    appt = Appointment.query.get_or_404(appt_id)

    send_sms(
        appt.phone,
        f"Hi {appt.name}, thanks for visiting Cherbon Waters.\n\n"
        f"Ready for the next step?\n\n"
        f"View our Wedding Information Portal, T&Cs and "
        f"booking information here:\n\n"
        f"https://cherbonapp.click/weddings/viewing-pack"
    )
    appt.status = 'quoted'

    db.session.commit()

    flash(
        "Follow-up SMS sent.",
        "success"
    )

    return redirect(
        url_for('hub.appointments_admin')
    )

# ---------------------------------------------------------
# ADMIN: ADD SLOT
# ---------------------------------------------------------

@hub_bp.route('/admin/slots/add', methods=['GET', 'POST'])
@login_required
def appointment_slot_add():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    if request.method == 'POST':

        slot = AppointmentSlot(
            appointment_date=datetime.strptime(
                request.form['appointment_date'],
                '%Y-%m-%d'
            ).date(),

            appointment_time=request.form['appointment_time'],

            is_available=('is_available' in request.form)
        )

        db.session.add(slot)
        db.session.commit()

        flash("Appointment slot created.", "success")

        return redirect(url_for('hub.appointment_slots'))

    return render_template(
        'weddings/appointment_slot_add.html',
        today=date.today().isoformat()
    )


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




@hub_bp.route(
    '/admin/appointments/<int:appt_id>/notes',
    methods=['POST']
)
@login_required
def appointment_save_notes(appt_id):

    if current_user.role not in ['admin', 'management']:
        abort(403)

    appt = Appointment.query.get_or_404(appt_id)

    appt.staff_notes = request.form.get(
        'staff_notes',
        ''
    )

    db.session.commit()

    flash(
        "Staff notes saved.",
        "success"
    )

    return redirect(
        request.referrer or
        url_for('hub.appointments_admin')
    )

@hub_bp.route('/admin/appointments/pipeline')
@login_required
def appointments_pipeline():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    appts = Appointment.query.filter(
        Appointment.status == 'quoted'
    ).order_by(
        Appointment.created_at.desc()
    ).all()

    return render_template(
        'weddings/appointments_pipeline.html',
        appointments=appts
    )

@hub_bp.route('/admin/appointments/archive')
@login_required
def appointments_archive():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    appts = Appointment.query.filter(
        Appointment.status == 'closed'
    ).order_by(
        Appointment.created_at.desc()
    ).all()

    return render_template(
        'weddings/appointments_archive.html',
        appointments=appts
    )


@hub_bp.route(
    '/admin/appointments/<int:appt_id>/send_reminder',
    methods=['POST']
)
@login_required
def appointment_send_reminder(appt_id):

    if current_user.role not in ['admin', 'management']:
        abort(403)

    appt = Appointment.query.get_or_404(appt_id)

    send_sms(
        appt.phone,
        f"Hi {appt.name}, just checking in following your "
        f"Cherbon Waters viewing. If you'd like to discuss "
        f"available dates, pricing or next steps, please "
        f"contact Jenny on 0405 184 280."
    )

    flash(
        "Reminder SMS sent.",
        "success"
    )

    return redirect(
        url_for('hub.appointments_pipeline')
    )

@hub_bp.route(
    '/admin/appointments/<int:appt_id>/close',
    methods=['POST']
)
@login_required
def appointment_close(appt_id):

    if current_user.role not in ['admin', 'management']:
        abort(403)

    appt = Appointment.query.get_or_404(appt_id)

    appt.status = 'closed'

    appt.close_reason = request.form.get(
        'close_reason',
        'Other'
    )

    db.session.commit()

    flash(
        "Lead closed.",
        "success"
    )

    return redirect(
        url_for('hub.appointments_pipeline')
    )

@hub_bp.route('/admin/appointment-templates')
@login_required
def appointment_templates():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    templates = AppointmentTemplate.query.order_by(
        AppointmentTemplate.weekday.asc(),
        AppointmentTemplate.appointment_time.asc()
    ).all()

    return render_template(
        'weddings/appointment_templates.html',
        templates=templates
    )

@hub_bp.route(
    '/admin/appointment-templates/add',
    methods=['POST']
)
@login_required
def appointment_template_add():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    t = AppointmentTemplate(
        weekday=int(request.form['weekday']),
        appointment_time=request.form['appointment_time']
    )

    db.session.add(t)
    db.session.commit()

    flash(
        "Template added.",
        "success"
    )

    return redirect(
        url_for('hub.appointment_templates')
    )

@hub_bp.route(
    '/admin/appointment-templates/<int:template_id>/delete',
    methods=['POST']
)
@login_required
def appointment_template_delete(template_id):

    if current_user.role not in ['admin', 'management']:
        abort(403)

    t = AppointmentTemplate.query.get_or_404(
        template_id
    )

    db.session.delete(t)

    db.session.commit()

    flash(
        "Template deleted.",
        "info"
    )

    return redirect(
        url_for('hub.appointment_templates')
    )


# ---------------------------------------------------------
# VIEWING APPOINTMENTS (PUBLIC + ADMIN)
# ---------------------------------------------------------

@hub_bp.route('/appointments', methods=['GET'])
def appointments_public():

    notice_cutoff = datetime.now() + timedelta(hours=15)

    all_slots = AppointmentSlot.query.filter(
        AppointmentSlot.is_available == True
    ).order_by(
        AppointmentSlot.appointment_date.asc(),
        AppointmentSlot.appointment_time.asc()
    ).all()

    slots = []

    for slot in all_slots:

        slot_dt = datetime.strptime(
            f"{slot.appointment_date} {slot.appointment_time}",
            "%Y-%m-%d %H:%M"
        )

        if slot_dt >= notice_cutoff:
            slots.append(slot)

    return render_template(
        'weddings/appointments_public.html',
        slots=slots
    )


@hub_bp.route('/appointments-v2')
def appointments_public_v2():

    templates = AppointmentTemplate.query.filter_by(
        is_active=True
    ).order_by(
        AppointmentTemplate.weekday,
        AppointmentTemplate.appointment_time
    ).all()

    today = date.today()

    end_date = today + timedelta(days=28)

    notice_cutoff = datetime.now() + timedelta(hours=15)

    generated_slots = []

    current_day = today

    while current_day <= end_date:

        for template in templates:

            if current_day.weekday() != template.weekday:
                continue

            blocked = AppointmentBlockout.query.filter(
                AppointmentBlockout.start_date <= current_day,
                AppointmentBlockout.end_date >= current_day
            ).first()

            if blocked:
                continue

            slot_blocked = AppointmentSlotBlockout.query.filter_by(
                appointment_date=current_day,
                appointment_time=template.appointment_time
            ).first()

            if slot_blocked:
                continue

            booking = Appointment.query.filter_by(
                day=current_day,
                time_label=template.appointment_time
            ).first()

            if booking:
                continue

            slot_dt = datetime.combine(
                current_day,
                datetime.strptime(
                    template.appointment_time,
                    "%H:%M"
                ).time()
            )

            if slot_dt < notice_cutoff:
                continue

            generated_slots.append(
                {
                    "id": 0,
                    "appointment_date": current_day,
                    "appointment_time": template.appointment_time
                }
            )

        current_day += timedelta(days=1)

    return render_template(
        'weddings/appointments_public_v2.html',
        slots=generated_slots
    )

@hub_bp.route('/appointments/book/<int:slot_id>',
              methods=['GET', 'POST'])
def book_viewing(slot_id):

    slot = AppointmentSlot.query.get_or_404(slot_id)

    if not slot.is_available:

        flash(
            "Sorry, this viewing appointment is no longer available.",
            "warning"
        )

        return redirect(
            url_for('hub.appointments_public')
        )

    if request.method == 'POST':

        appt = Appointment(
            name=request.form['name'],
            email=request.form['email'],
            phone=request.form['phone'],
            day=slot.appointment_date,
            time_label=slot.appointment_time
        )

        db.session.add(appt)

        slot.is_available = False

        db.session.commit()

        try:

            h, m = map(int, slot.appointment_time.split(':'))

            formatted_time = (
                f"{((h - 1) % 12) + 1}:{m:02d} "
                f"{'AM' if h < 12 else 'PM'}"
            )

            send_sms(
                appt.phone,
                f"Hi {appt.name}, your Cherbon Waters viewing is confirmed "
                f"for {slot.appointment_date.strftime('%d/%m/%Y')} "
                f"at {formatted_time}."
            )

        except Exception as e:

            print(f"SMS FAILED: {e}")

        return render_template(
            'weddings/appointments_thankyou.html',
            appointment=appt
        )

    return render_template(
        'weddings/book_viewing.html',
        slot=slot
    )


@hub_bp.route(
    '/appointments/book-v2',
    methods=['GET', 'POST']
)
def book_viewing_v2():

    appointment_date = datetime.strptime(
        request.args['appointment_date'],
        '%Y-%m-%d'
    ).date()

    appointment_time = request.args[
        'appointment_time'
    ]

    existing = Appointment.query.filter_by(
        day=appointment_date,
        time_label=appointment_time
    ).first()

    if existing:

        flash(
            "Sorry, this viewing appointment is no longer available.",
            "warning"
        )

        return redirect(
            url_for('hub.appointments_public_v2')
        )

    if request.method == 'POST':

        appt = Appointment(
            name=request.form['name'],
            email=request.form['email'],
            phone=request.form['phone'],
            day=appointment_date,
            time_label=appointment_time
        )

        db.session.add(appt)
        db.session.commit()

        try:

            h, m = map(
                int,
                appointment_time.split(':')
            )

            formatted_time = (
                f"{((h - 1) % 12) + 1}:{m:02d} "
                f"{'AM' if h < 12 else 'PM'}"
            )

            send_sms(
                appt.phone,
                f"Hi {appt.name}, your Cherbon Waters "
                f"viewing is confirmed for "
                f"{appointment_date.strftime('%d/%m/%Y')} "
                f"at {formatted_time}."
            )

        except Exception as e:

            print(f"SMS FAILED: {e}")

        return render_template(
            'weddings/appointments_thankyou.html',
            appointment=appt
        )

    return render_template(
        'weddings/book_viewing_v2.html',
        appointment_date=appointment_date,
        appointment_time=appointment_time
    )


@hub_bp.route('/admin/appointment-exceptions')
@login_required
def appointment_exceptions():

    if current_user.role not in ['admin', 'management']:
        abort(403)

    date_blockouts = AppointmentBlockout.query.order_by(
        AppointmentBlockout.start_date.asc()
    ).all()

    time_blockouts = AppointmentSlotBlockout.query.order_by(
        AppointmentSlotBlockout.appointment_date.asc(),
        AppointmentSlotBlockout.appointment_time.asc()
    ).all()

    return render_template(
        'weddings/appointment_exceptions.html',
        date_blockouts=date_blockouts,
        time_blockouts=time_blockouts
    )