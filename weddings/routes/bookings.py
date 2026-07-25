from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import datetime
from app import db
from weddings.models import Wedding

bookings_bp = Blueprint('bookings', __name__, url_prefix='/weddings')

# ---------------------------------------------------------
# WEDDING CALENDAR (ADMIN ONLY)
# ---------------------------------------------------------
@bookings_bp.route('/calendar')
def wedding_calendar():
    weddings = Wedding.query.order_by(Wedding.wedding_date.asc()).all()

    rows = []
    for w in weddings:
        rows.append({
            "id": w.id,
            "date": w.wedding_date.strftime('%d %b %Y'),
            "bride": w.bride_name,
            "groom": w.groom_name,
            "status": w.status,
            "notes": w.notes
        })

    return render_template('weddings/calendar.html', weddings=rows)


# ---------------------------------------------------------
# ADD WEDDING (MANUAL HOLD / TENTATIVE)
# ---------------------------------------------------------
@bookings_bp.route('/add', methods=['GET', 'POST'])
def add_wedding():
    if request.method == 'POST':
        bride = request.form.get('bride_name')
        groom = request.form.get('groom_name')
        date_str = request.form.get('wedding_date')
        notes = request.form.get('notes', '')
        status = request.form.get('status', 'hold')  # default hold

        if not date_str:
            flash("Wedding date is required", "danger")
            return redirect(url_for('bookings.add_wedding'))

        wedding_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        new_wedding = Wedding(
            bride_name=bride,
            groom_name=groom,
            wedding_date=wedding_date,
            notes=notes,
            status=status,
            booking_source='manual'
        )

        db.session.add(new_wedding)
        db.session.commit()

        return redirect(url_for('bookings.wedding_calendar'))

    return render_template('weddings/add.html')


# ---------------------------------------------------------
# EDIT WEDDING
# ---------------------------------------------------------
@bookings_bp.route('/edit/<int:wedding_id>', methods=['GET', 'POST'])
def edit_wedding(wedding_id):
    wedding = Wedding.query.get_or_404(wedding_id)

    if request.method == 'POST':
        wedding.bride_name = request.form.get('bride_name')
        wedding.groom_name = request.form.get('groom_name')
        wedding.notes = request.form.get('notes')

        date_str = request.form.get('wedding_date')
        if date_str:
            wedding.wedding_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        wedding.status = request.form.get('status', wedding.status)

        db.session.commit()
        return redirect(url_for('bookings.wedding_calendar'))

    return render_template('weddings/edit.html', wedding=wedding)


# ---------------------------------------------------------
# DELETE WEDDING
# ---------------------------------------------------------
@bookings_bp.route('/delete/<int:wedding_id>')
def delete_wedding(wedding_id):
    wedding = Wedding.query.get_or_404(wedding_id)
    db.session.delete(wedding)
    db.session.commit()
    return redirect(url_for('bookings.wedding_calendar'))


# ---------------------------------------------------------
# STATUS CHANGE ROUTES
# ---------------------------------------------------------
@bookings_bp.route('/status/<int:wedding_id>/<string:new_status>')
def change_status(wedding_id, new_status):
    wedding = Wedding.query.get_or_404(wedding_id)

    valid_statuses = ['hold', 'booked', 'tentative', 'cancelled', 'postponed', 'archived']
    if new_status not in valid_statuses:
        flash("Invalid status", "danger")
        return redirect(url_for('bookings.wedding_calendar'))

    wedding.status = new_status

    # Timestamp logic
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
# MERGE MANUAL → JOTFORM T&C SUBMISSION
# ---------------------------------------------------------
@bookings_bp.route('/merge_tnc/<int:wedding_id>/<string:submission_id>')
def merge_tnc(wedding_id, submission_id):
    wedding = Wedding.query.get_or_404(wedding_id)

    wedding.tnc_submission_id = submission_id
    wedding.booking_source = 'jotform_tnc'
    wedding.status = 'booked'

    db.session.commit()
    return redirect(url_for('bookings.wedding_calendar'))
