from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from flask_login import login_required, current_user
from datetime import datetime
from extensions import db
from auth.permissions import admin_required, is_coordinator, is_caterer

from weddings.models import (
    Wedding,
    WeddingStaffAssignment,
    JobListInstance,
    JobListMaster,
    WeddingTimeline,
    WeddingDetailLibrary,
    Caterer
)

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
        Wedding.booking_status.notin_(["archived", "cancelled"])
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

