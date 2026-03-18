from extensions import db
from datetime import datetime


class Client(db.Model):
    __tablename__ = 'clients'

    # --- Identity ---
    client_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String)

    # --- Guardian / Family ---
    guardian_name = db.Column(db.String)
    guardian_contact = db.Column(db.String)   # legacy field, safe to keep
    guardian2_name = db.Column(db.String)
    mobile2 = db.Column(db.String)
    email_guardian2 = db.Column(db.String)

    # --- Contact ---
    mobile = db.Column(db.String)
    email_primary = db.Column(db.String)
    email_secondary = db.Column(db.String)

    # --- Rider Details ---
    age = db.Column(db.Integer)
    weight_kg = db.Column(db.Integer, nullable=True)
    height_cm = db.Column(db.Integer, nullable=True)

    # --- Admin / Compliance ---
    disclaimer = db.Column(db.Integer)
    invoice_required = db.Column(db.Boolean, default=False)
    jotform_submission_id = db.Column(db.String)

    # --- NDIS ---
    ndis_number = db.Column(db.String)
    ndis_code = db.Column(db.String)

    # --- Notes ---
    notes = db.Column(db.String)
    notes2 = db.Column(db.String)

class LessonInvite(db.Model):
    __tablename__ = 'lesson_invites'

    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(db.Integer, nullable=True)
    token = db.Column(db.String, nullable=False)
    mobile = db.Column(db.String, nullable=False)
    riders_requested = db.Column(db.Integer, nullable=False)
    cost_per_person = db.Column(db.Float, nullable=False)
    time_frame = db.Column(db.String, nullable=True)
    lesson_type = db.Column(db.String, nullable=True)
    group_priv = db.Column(db.String(20))   # 🔥 ADD THIS LINE
    status = db.Column(db.String, nullable=False)
    lesson_date = db.Column(db.Date, nullable=False)

class DisclaimerState(db.Model):
    __tablename__ = 'disclaimer_state'

    id = db.Column(db.Integer, primary_key=True)
    max_disclaimer_number = db.Column(db.Integer, nullable=False, default=0)


class IncomingSubmission(db.Model):
    __tablename__ = 'incoming_submissions'

    id = db.Column(db.Integer, primary_key=True)

    # PRIMARY DEDUPE KEY (must always be set)
    submission_id = db.Column(db.String, index=True, nullable=False)

    # JotForm form ID
    form_id = db.Column(db.String, nullable=False)

    # Raw JSON payload
    raw_payload = db.Column(db.JSON, nullable=False)

    # REAL JotForm timestamp (created_at)
    received_at = db.Column(db.DateTime, index=True, nullable=False)

    # Processing flags
    processed = db.Column(db.Boolean, default=False)
    processed_at = db.Column(db.DateTime)

    # Secondary dedupe
    unique_hash = db.Column(db.String(64), index=True)

    # Invite flow
    needs_client_match = db.Column(db.Boolean, default=False)

    # Disclaimer flow
    ignored = db.Column(db.Boolean, default=False)
    
    # JotForm ID (same as submission_id)
    jotform_id = db.Column(db.String, index=True)



class TeacherTime(db.Model):
    __tablename__ = 'teacher_time'
    id = db.Column(db.Integer, primary_key=True)
    teacher_key = db.Column(db.String(64), nullable=False, index=True)  # e.g. "Teacher 1"
    weekday = db.Column(db.Integer, nullable=False, index=True)         # 0=Mon … 6=Sun
    time = db.Column(db.String(16), nullable=False)                     # "08:00"

class Horse(db.Model):
    __tablename__ = 'horses'
    horse_id = db.Column(db.Integer, primary_key=True)
    horse = db.Column(db.Text)
    orderpdk = db.Column(db.Integer)   # ⭐ ADD THIS


class Teacher(db.Model):
    __tablename__ = 'teachers'
    teacher_id = db.Column(db.Integer, primary_key=True)
    teacher = db.Column(db.String(100), nullable=False, unique=True)


class Time(db.Model):
    __tablename__ = 'times'
    time_id = db.Column(db.Integer, primary_key=True)
    timerange = db.Column(db.Text, unique=True)   # ✅ enforce one row per timerange
    length = db.Column(db.Text)

class BlockoutDate(db.Model):
    __tablename__ = 'blockout_dates'
    id = db.Column(db.Integer, primary_key=True)
    block_date = db.Column(db.Date, nullable=False)
    reason = db.Column(db.String(255))

class BlockoutRange(db.Model):
    __tablename__ = 'blockout_ranges'
    id = db.Column(db.Integer, primary_key=True)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    reason = db.Column(db.String(255))

class Lesson(db.Model):
    __tablename__ = 'lessons'

    lesson_id = db.Column(db.Integer, primary_key=True)
    lesson_date = db.Column(db.Date)
    time_frame = db.Column(db.String)
    client = db.Column(db.String)
    horse = db.Column(db.String)
    adjust = db.Column(db.Integer, default=0)   # <— NEW FIELD
    carry_fwd = db.Column(db.Float)
    payment = db.Column(db.Float)
    price_pl = db.Column(db.Float)
    attendance = db.Column(db.String)
    balance = db.Column(db.Float)
    lesson_notes = db.Column(db.Text)
    lesson_type = db.Column(db.String)
    group_priv = db.Column(db.String)
    blockends = db.Column(db.Date)
    lesson_no = db.Column(db.Text)
    freq = db.Column(db.String(2))   # "S", "F", "W"

class TeacherHorse(db.Model):
    __tablename__ = 'teacher_horse'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    block_key = db.Column(db.String(50), nullable=False)
    horse1 = db.Column(db.String(100))
    horse2 = db.Column(db.String(100))

class TeacherSlot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lesson_date = db.Column(db.Date, nullable=False)
    slot_number = db.Column(db.Integer, nullable=False)
    teacher_name = db.Column(db.String(50))


class Users(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    active = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), default="management")


class LessonBlockTag(db.Model):
    __tablename__ = 'lesson_block_tags'
    id = db.Column(db.Integer, primary_key=True)
    lesson_date = db.Column(db.Date, nullable=False)
    time_range = db.Column(db.String(20), nullable=False)  # e.g. '08:00 - 09:00'
    lesson_type = db.Column(db.String(50), nullable=False)
    group_priv = db.Column(db.String(20))
    teacher_tags = db.Column(db.Text)  # e.g. 'T1,T2,T4'

    # New fields for explicit teacher names
    teacher1 = db.Column(db.String(50))  # Name for T1
    teacher2 = db.Column(db.String(50))  # Name for T2
    teacher3 = db.Column(db.String(50))  # Name for T3
    teacher4 = db.Column(db.String(50))  # Name for T4
    teacher5 = db.Column(db.String(50))  # Name for T5



    # ❌ Do NOT define relationships unless you have foreign keys like client_id, horse_id, etc.
    # Remove these:
    # client = db.relationship('Client', backref='lessons')
    # horse = db.relationship('Horse', backref='lessons')
    # teacher = db.relationship('Teacher', backref='lessons')
    # time = db.relationship('Time', backref='lessons')

