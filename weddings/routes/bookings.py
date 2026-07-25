from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from datetime import datetime
from extensions import db
from weddings.models import Wedding
from flask_login import login_required, current_user

# ---------------------------------------------------------
# DEFINE BLUEPRINT FIRST
# ---------------------------------------------------------
bookings_bp = Blueprint('bookings', __name__, url_prefix='/weddings')

# ---------------------------------------------------------
# LOCK DOWN ALL /weddings ROUTES
# (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.before_request
@login_required
def protect_weddings():
    if current_user.role not in ['admin', 'management', 'manager', 'coordinator']:
        abort(403)


# ---------------------------------------------------------
# WEDDING CALENDAR (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.route('/calendar')
def wedding_calendar():
    show_archived = request.args.get('show_archived', '0') == '1'

    query = Wedding.query.order_by(Wedding.wedding_date.asc())

    if not show_archived:
        query = query.filter(Wedding.status != 'archived')

    weddings = query.all()

    rows = []
    for w in weddings:
        rows.append({
            "id": w.id,
            "date": w.wedding_date.strftime('%a %d-%m-%Y'),
            "bride": w.bride_name,
            "groom": w.groom_name,
            "status": w.status,
            "notes": w.other_information,
            "event_type": w.event_type,
            "bride_mobile": w.bride_mobile,
            "service": w.service,
            "est_guests": w.est_guests
        })

    return render_template('weddings/calendar.html',
                           weddings=rows,
                           show_archived=show_archived)


# ---------------------------------------------------------
# ADD WEDDING (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.route('/add', methods=['GET', 'POST'])
def add_wedding():
    if request.method == 'POST':
        bride = request.form.get('bride_name')
        groom = request.form.get('groom_name')
        date_str = request.form.get('wedding_date')
        other_information = request.form.get('other_information', '')
        status = request.form.get('status', 'hold')

        # NEW FIELDS
        event_type = request.form.get('event_type')
        bride_mobile = request.form.get('bride_mobile')
        service = request.form.get('service')
        est_guests = request.form.get('est_guests')

        if not date_str:
            flash("Wedding date is required", "danger")
            return redirect(url_for('bookings.add_wedding'))

        wedding_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        new_wedding = Wedding(
            bride_name=bride,
            groom_name=groom,
            wedding_date=wedding_date,
            other_information=other_information,
            status=status,
            booking_source='manual',

            event_type=event_type,
            bride_mobile=bride_mobile,
            service=service,
            est_guests=est_guests
        )

        db.session.add(new_wedding)
        db.session.commit()

        return redirect(url_for('bookings.wedding_calendar'))

    return render_template('weddings/add.html')


# ---------------------------------------------------------
# EDIT WEDDING (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.route('/edit/<int:wedding_id>', methods=['GET', 'POST'])
def edit_wedding(wedding_id):
    wedding = Wedding.query.get_or_404(wedding_id)

    if request.method == 'POST':
        wedding.bride_name = request.form.get('bride_name')
        wedding.groom_name = request.form.get('groom_name')
        wedding.other_information = request.form.get('other_information')

        # NEW FIELDS
        wedding.event_type = request.form.get('event_type')
        wedding.bride_mobile = request.form.get('bride_mobile')
        wedding.service = request.form.get('service')
        wedding.est_guests = request.form.get('est_guests')

        date_str = request.form.get('wedding_date')
        if date_str:
            wedding.wedding_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        wedding.status = request.form.get('status', wedding.status)

        db.session.commit()
        return redirect(url_for('bookings.wedding_calendar'))

    return render_template('weddings/edit.html', wedding=wedding)


# ---------------------------------------------------------
# DELETE WEDDING (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.route('/delete/<int:wedding_id>')
def delete_wedding(wedding_id):
    wedding = Wedding.query.get_or_404(wedding_id)
    db.session.delete(wedding)
    db.session.commit()
    return redirect(url_for('bookings.wedding_calendar'))


# ---------------------------------------------------------
# STATUS CHANGE (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.route('/status/<int:wedding_id>/<string:new_status>')
def change_status(wedding_id, new_status):
    wedding = Wedding.query.get_or_404(wedding_id)

    valid_statuses = ['hold', 'booked', 'tentative', 'cancelled', 'postponed', 'archived']
    if new_status not in valid_statuses:
        flash("Invalid status", "danger")
        return redirect(url_for('bookings.wedding_calendar'))

    wedding.status = new_status

    now = datetime.utcnow()
    if new_status == 'cancelled':
        wedding.cancelled_at = now
    elif new_status == 'postponed':
        wedding.postponed_at = now
    elif new_status == 'archived':
        wedding.archived_at = now

    db.session.commit()
    return redirect(url_for('bookings.wedding_calendar'))


# ---------------------------------------------------------
# MERGE MANUAL → JOTFORM T&C (Admin / Management / Manager / Coordinator)
# ---------------------------------------------------------
@bookings_bp.route('/merge_tnc/<int:wedding_id>/<string:submission_id>')
def merge_tnc(wedding_id, submission_id):
    wedding = Wedding.query.get_or_404(wedding_id)

    wedding.tnc_submission_id = submission_id
    wedding.booking_source = 'jotform_tnc'
    wedding.status = 'booked'

    db.session.commit()
    return redirect(url_for('bookings.wedding_calendar'))
