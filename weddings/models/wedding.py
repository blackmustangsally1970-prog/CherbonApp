from datetime import datetime
from extensions import db

class Wedding(db.Model):
    __tablename__ = 'wedding'

    id = db.Column(db.Integer, primary_key=True)

    # ============================
    # CLIENT INFO (CSV + JotForm)
    # ============================
    bride_name = db.Column(db.String(120))
    bride_mobile = db.Column(db.String(50))
    bride_email = db.Column(db.String(120))

    groom_name = db.Column(db.String(120))
    groom_mobile = db.Column(db.String(50))
    groom_email = db.Column(db.String(120))

    client_email = db.Column(db.String(120))   # legacy
    client_phone = db.Column(db.String(50))    # legacy

    # ============================
    # EVENT DETAILS
    # ============================
    wedding_date = db.Column(db.Date, nullable=False)
    event_type = db.Column(db.String(120))     # CO / IND / WR / etc
    options = db.Column(db.String(255))
    service = db.Column(db.String(255))
    est_guests = db.Column(db.Integer)
    start_time = db.Column(db.String(50))
    other_information = db.Column(db.Text)

    # ============================
    # SUBMISSION METADATA
    # ============================
    submission_date = db.Column(db.Date)
    price_version = db.Column(db.String(50))
    tnc_version = db.Column(db.String(50))

    # ============================
    # BOOKING STATUS
    # ============================
    status = db.Column(db.String(50), default='hold')  
    booking_source = db.Column(db.String(50))          # manual / jotform_tnc

    # ============================
    # JOTFORM LINKS
    # ============================
    tnc_submission_id = db.Column(db.String(120))
    menu_submission_id = db.Column(db.String(120))

    # ============================
    # RELATIONSHIPS (Hub)
    # ============================
    staff_assignments = db.relationship(
        'WeddingStaffAssignment',
        backref='wedding',
        cascade="all, delete-orphan"
    )

    job_lists = db.relationship(
        'JobListInstance',
        backref='wedding',
        cascade="all, delete-orphan"
    )

    timeline = db.relationship(
        'WeddingTimeline',
        backref='wedding',
        cascade="all, delete-orphan"
    )
