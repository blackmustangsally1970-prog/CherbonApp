from extensions import db
from datetime import datetime
from app import db



class DailyEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    fy = db.Column(db.String(9), nullable=False)  # "2024-2025"
    field1 = db.Column(db.String(255))
    field2 = db.Column(db.String(255))

    __table_args__ = (
        db.UniqueConstraint('date', 'fy', name='uq_daily_date_fy'),
    )


class Term(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    term_number = db.Column(db.Integer, nullable=False)  # 1–4
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    week_pattern = db.Column(db.String(20), default="Sun-Sat")  # or "Mon-Sun"
    weeks = db.Column(db.Integer, default=10)
    active = db.Column(db.Boolean, default=False)

    @property
    def label(self):
        return f"Term {self.term_number} {self.year}"



class GroupPricing(db.Model):
    __tablename__ = "group_pricing"

    id = db.Column(db.Integer, primary_key=True)
    group_priv = db.Column(db.String(10), unique=True, nullable=False)

    weekly_price = db.Column(db.Float, default=0)
    fortnightly_price = db.Column(db.Float, default=0)
    full_price = db.Column(db.Float, default=0)

    two_course_weekly = db.Column(db.Float, default=0)
    two_course_fortnightly = db.Column(db.Float, default=0)
    two_course_full = db.Column(db.Float, default=0)

    sibling_weekly = db.Column(db.Float, default=0)
    sibling_fortnightly = db.Column(db.Float, default=0)
    sibling_full = db.Column(db.Float, default=0)


class CourseFormSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rider_name = db.Column(db.String(120))
    courseno = db.Column(db.String(50))
    ftor = db.Column(db.String(50))
    horse_1 = db.Column(db.String(120))
    horse_2 = db.Column(db.String(120))
    horse_3 = db.Column(db.String(120))
    notes = db.Column(db.String(500))   # ← ADD THIS
    submitted_at = db.Column(db.DateTime)



class Employee(db.Model):
    __tablename__ = "employees"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    setup_code = db.Column(db.String(20), unique=True)
    phone = db.Column(db.String(30))  # <-- NEW FIELD
    pin_hash = db.Column(db.String(200))
    active = db.Column(db.Boolean, default=True)

    # Lockout system
    pin_failures = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)


class EmployeeHours(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"))
    date = db.Column(db.Date)
    sign_in = db.Column(db.DateTime)
    break_start = db.Column(db.DateTime)
    break_end = db.Column(db.DateTime)
    sign_out = db.Column(db.DateTime)
    auto_prompted = db.Column(db.Boolean, default=False)
    employee = db.relationship("Employee", backref="hours")
    submitted_at = db.Column(db.DateTime)

    corrected = db.Column(db.Boolean, default=False)
    corrected_at = db.Column(db.DateTime)
    corrected_by = db.Column(db.Integer)  # admin user ID


class GiftVoucherSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    submission_id = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    purchaser_name = db.Column(db.String(120))
    purchaser_email = db.Column(db.String(120))
    purchaser_mobile = db.Column(db.String(40))

    recipient_name = db.Column(db.String(120))
    ridden_before = db.Column(db.String(20))

    voucher_choice = db.Column(db.String(120))
    quantity = db.Column(db.String(10))
    ride_type = db.Column(db.String(200))
    credit_amount = db.Column(db.String(20))

    voucher_number = db.Column(db.String(32))
    amount_payable = db.Column(db.String(20))

    message_to_recipient = db.Column(db.Text)

    ignored = db.Column(db.Boolean, default=False)
    processed = db.Column(db.Boolean, default=False)
    processed_at = db.Column(db.DateTime)



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
    disclaimer_date = db.Column(db.Date, nullable=True)
    disclaimer_source = db.Column(db.String, nullable=True)
    disclaimer_version = db.Column(db.String, nullable=True)
    invoice_required = db.Column(db.Boolean, default=False)
    jotform_submission_id = db.Column(db.String)

    # --- NDIS ---
    ndis_number = db.Column(db.String)
    ndis_code = db.Column(db.String)

    # --- Notes ---
    notes = db.Column(db.String)
    notes2 = db.Column(db.String)

class JotformFetchState(db.Model):
    __tablename__ = 'jotform_fetch_state'

    id = db.Column(db.Integer, primary_key=True)
    last_fetched_submission_id = db.Column(db.String, default="0")
    last_fetched_timestamp = db.Column(db.DateTime)



class UpgradeItem(db.Model):
    __tablename__ = 'upgrade_items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)



class WeddingStaffUnavailability(db.Model):
    __tablename__ = 'wedding_staff_unavailability'

    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey('wedding_staff.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    reason = db.Column(db.Text)

    staff = db.relationship('WeddingStaff', backref='unavailability')


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

class WeeklyEvent(db.Model):
    __tablename__ = "weekly_events"

    id = db.Column(db.Integer, primary_key=True)
    week_start = db.Column(db.Date, nullable=False)
    fy = db.Column(db.String, nullable=False)

class TrailRideSubmission(db.Model):
    __tablename__ = 'trail_ride_submissions'

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.String(255), unique=True, nullable=False)
    form_id = db.Column(db.String(255), nullable=False)
    raw_payload = db.Column(db.JSON, nullable=False)
    received_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    processed = db.Column(db.Boolean, nullable=False, default=False)
    ignored = db.Column(db.Boolean, nullable=False, default=False)
    needs_client_match = db.Column(db.Boolean, nullable=False, default=False)
    jotform_id = db.Column(db.String(255))



class TeacherBlock(db.Model):
    __tablename__ = "teacher_blocks"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String, nullable=False)
    block_key = db.Column(db.String, nullable=False)

    horse = db.Column(db.String, nullable=True)
    teacher_name = db.Column(db.String, nullable=True)
    notes = db.Column(db.String, nullable=True)

    # Optional: timestamp for debugging
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class LessonTeacherTag(db.Model):
    __tablename__ = "lesson_teacher_tags"

    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(db.Integer, unique=True, nullable=False)
    lesson_date = db.Column(db.Date, nullable=False)

    t1 = db.Column(db.Boolean, default=False)
    t2 = db.Column(db.Boolean, default=False)
    t3 = db.Column(db.Boolean, default=False)
    t4 = db.Column(db.Boolean, default=False)
    t5 = db.Column(db.Boolean, default=False)

    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


class CourseReference(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    course_code = db.Column(db.String(10), unique=True, nullable=False)
    display_label = db.Column(db.String(120), nullable=False)

    day_of_week = db.Column(db.String(10), nullable=False)

    # NEW — replaces start_time + end_time
    timerange = db.Column(db.String(20), nullable=False)

    lesson_type = db.Column(db.String(20), nullable=False)
    group_priv = db.Column(db.String(5), nullable=False)

    sort_order = db.Column(db.Integer, default=0)
    active = db.Column(db.Boolean, default=True)


class Wedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    notes = db.Column(db.Text)
    category = db.Column(db.String(10), default='WR')
    pax = db.Column(db.Integer)
    time = db.Column(db.String(50))
    service1 = db.Column(db.String(100))   # ← single editable 1st Service field

    assignments = db.relationship(
        'WeddingAssignment',
        backref='wedding',
        cascade="all, delete-orphan"
    )


class WeddingStaff(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)

    assignments = db.relationship('WeddingAssignment', backref='staff', cascade="all, delete-orphan")


class WeddingAssignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)
    staff_id = db.Column(db.Integer, db.ForeignKey('wedding_staff.id'), nullable=False)



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


class GeneralEnquirySubmission(db.Model):
    __tablename__ = "general_enquiry_submissions"

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.String(64), unique=True, nullable=False)

    created_at = db.Column(db.DateTime, nullable=False)

    rider_name = db.Column(db.String(120))
    rider_age = db.Column(db.Integer)
    rider_height_cm = db.Column(db.Integer)
    rider_weight_kg = db.Column(db.Integer)

    email_address = db.Column(db.String(120))
    mobile_phone = db.Column(db.String(40))
    processed = db.Column(db.Boolean, default=False)
    processed_at = db.Column(db.DateTime)
    comments = db.Column(db.Text)
    ignored = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<GeneralEnquirySubmission {self.submission_id}>"



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
    block_key = db.Column(db.String)   # <— ADD THIS
    client = db.Column(db.String)
    horse = db.Column(db.String)
    adjust = db.Column(db.Integer, default=0)
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
    freq = db.Column(db.String(2))
    voucher_number = db.Column(db.String(32))


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


class TeacherGridOverride(db.Model):
    __tablename__ = 'teacher_grid_overrides'

    id = db.Column(db.Integer, primary_key=True)
    override_date = db.Column(db.Date, nullable=False)
    time_label = db.Column(db.String(20), nullable=False)   # e.g. "10:30"
    teacher_index = db.Column(db.Integer, nullable=False)   # 1–5
    state = db.Column(db.Integer, nullable=False)           # 0=green, 1=red, 2=break
    updated_at = db.Column(db.DateTime, server_default=db.func.now())

    __table_args__ = (
        db.UniqueConstraint('override_date', 'time_label', 'teacher_index'),
    )




    # ❌ Do NOT define relationships unless you have foreign keys like client_id, horse_id, etc.
    # Remove these:
    # client = db.relationship('Client', backref='lessons')
    # horse = db.relationship('Horse', backref='lessons')
    # teacher = db.relationship('Teacher', backref='lessons')
    # time = db.relationship('Time', backref='lessons')

