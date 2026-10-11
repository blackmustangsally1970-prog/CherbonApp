from datetime import datetime
from extensions import db

class Wedding(db.Model):
    __tablename__ = 'wedding'

    id = db.Column(db.Integer, primary_key=True)

    # ============================
    # CLIENT INFO
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
    start_time = db.Column(db.String(50))
    event_type = db.Column(db.String(120))
    options = db.Column(db.String(255))
    service = db.Column(db.String(255))
    est_guests = db.Column(db.String(50))
    other_information = db.Column(db.Text)

    # ============================
    # HOLD SYSTEM
    # ============================
    is_hold = db.Column(db.Boolean, default=False)
    hold_name = db.Column(db.String(120))
    hold_notes = db.Column(db.Text)
    hold_created_at = db.Column(db.DateTime)
    hold_expires_at = db.Column(db.DateTime)

    # ============================
    # CATERER DETAILS
    # ============================
    caterer_id = db.Column(db.Integer, db.ForeignKey('caterer.id'))
    caterer = db.relationship("Caterer")

    # ============================
    # SUBMISSION METADATA
    # ============================
    submission_date = db.Column(db.DateTime)
    price_version = db.Column(db.String(50))
    tnc_version = db.Column(db.String(50))

    # ============================
    # BOOKING WORKFLOW
    # ============================
    booking_status = db.Column(db.String(50), default='pending_admin')
    admin_approved_at = db.Column(db.DateTime)

    # ============================
    # VENDOR NOTIFICATIONS
    # ============================
    caterer_notified_at = db.Column(db.DateTime)
    caterer_confirmed_at = db.Column(db.DateTime)

    decorator_notified_at = db.Column(db.DateTime)
    decorator_confirmed_at = db.Column(db.DateTime)

    coordinator_notified_at = db.Column(db.DateTime)
    coordinator_confirmed_at = db.Column(db.DateTime)

    staff_notified_at = db.Column(db.DateTime)
    staff_confirmed_at = db.Column(db.DateTime)

    # ============================
    # JOTFORM LINKS
    # ============================
    tnc_submission_id = db.Column(db.String(120))
    menu_submission_id = db.Column(db.String(120))

    # ============================
    # RELATIONSHIPS
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
