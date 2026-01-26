from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, make_response, after_this_request, flash, jsonify
)
from config import Config
from extensions import db
from collections import defaultdict
from sqlalchemy import func, text
from datetime import date, datetime, time
from models import (
    Lesson, Time, Client, Horse, Teacher,
    LessonBlockTag, TeacherTime, TeacherHorse,
    BlockoutDate, BlockoutRange, IncomingSubmission,
    LessonInvite
)
import secrets
import string
from urllib.parse import urlencode
import re
import subprocess
import tempfile
import os
import unicodedata
import json
import hashlib
import clicksend_client
from clicksend_client import SmsMessage
from clicksend_client.rest import ApiException
from functools import lru_cache


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(BASE_DIR, "lesson_logs")
os.makedirs(LOG_ROOT, exist_ok=True)

WEDDING_SMS_ROOT = os.path.join(BASE_DIR, "weddingsms")
os.makedirs(WEDDING_SMS_ROOT, exist_ok=True)

CSV_ROOT = os.path.join(BASE_DIR, "csv")
os.makedirs(CSV_ROOT, exist_ok=True)


@lru_cache(maxsize=1)
def get_static_clients():
    return db.session.query(Client).order_by(Client.full_name).all()

@lru_cache(maxsize=1)
def get_static_horses():
    return db.session.query(Horse).order_by(Horse.horse).all()

@lru_cache(maxsize=1)
def get_static_times():
    return db.session.query(Time).order_by(Time.timerange).all()

@lru_cache(maxsize=1)
def get_static_teachers():
    return db.session.query(Teacher).order_by(Teacher.teacher).all()

@lru_cache(maxsize=1)
def get_static_teacher_time():
    return db.session.query(TeacherTime).order_by(
        TeacherTime.teacher_key,
        TeacherTime.weekday,
        TeacherTime.time
    ).all()


def recalc_client_lessons_by_id(client_id):
    client_obj = Client.query.get(client_id)
    if not client_obj:
        return

    # Load ALL lessons for this client, oldest → newest
    lessons = (
        Lesson.query
        .filter(Lesson.client == client_obj.full_name)
        .order_by(
            db.func.date(Lesson.lesson_date).asc(),
            Lesson.lesson_id.asc()
        )
        .all()
    )

    bal = 0
    for l in lessons:
        att   = (l.attendance or '').strip().upper()
        pay   = l.payment or 0
        price = l.price_pl or 0

        chargeable = att in ['Y', 'N']

        if chargeable:
            new_balance = bal + pay - price
        else:
            new_balance = bal + pay

        l.balance = new_balance
        bal = new_balance

    db.session.commit()


def log_lesson_changes(changes, user="system"):
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_date = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_ROOT, f"lesson_changes_{log_date}.log")

    # Separate lesson and client keys
    lesson_ids = [lid for lid in changes if isinstance(lid, int)]
    client_ids = [int(str(k).split("_")[1]) for k in changes if str(k).startswith("client_")]

    lessons = db.session.query(Lesson).filter(Lesson.lesson_id.in_(lesson_ids)).all()
    clients = db.session.query(Client).filter(Client.client_id.in_(client_ids)).all()

    # 🔑 keep both client name and lesson date
    lesson_lookup = {
        lesson.lesson_id: (lesson.client, lesson.lesson_date)
        for lesson in lessons
    }
    client_lookup = {client.client_id: client.full_name for client in clients}

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] by {user}\n")
            for key in sorted(changes, key=lambda x: str(x)):
                if str(key).startswith("client_"):
                    cid = int(str(key).split("_")[1])
                    client = client_lookup.get(cid, f"Client {cid}")
                    lesson_date = ""   # no specific lesson date for client notes
                else:
                    client, lesson_date = lesson_lookup.get(key, (f"Lesson {key}", None))

                fields = changes[key]
                client_fmt = f"{client:<35}"
                date_fmt = lesson_date.strftime("%Y-%m-%d") if lesson_date else "—"

                parts = []
                if "payment" in fields:
                    old, new = fields["payment"]
                    if old != new:
                        parts.append(f"payment: '{old}' -> '{new}'")
                if "attendance" in fields:
                    old, new = fields["attendance"]
                    if old != new:
                        parts.append(f"attendance: '{old}' -> '{new}'")
                if "horse" in fields:
                    old, new = fields["horse"]
                    if old != new:
                        parts.append(f"horse: '{old}' -> '{new}'")
                if "notes" in fields:
                    old, new = fields["notes"]
                    if (old or '').strip() != (new or '').strip():
                        parts.append(f"notes: '{old}' -> '{new}'")

                if parts:
                    # ✅ now includes lesson date
                    f.write(f"  {client_fmt} | {date_fmt} | " + " | ".join(parts) + "\n")
            f.write("\n")
    except Exception as e:
        print(f"Log write failed: {e}")

def log_disclaimer_processed(names):
    from datetime import datetime
    log_date = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_path = os.path.join(LOG_ROOT, f"disclaimers_{log_date}.txt")



    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {', '.join(names)}\n")
    except Exception as e:
        print("Failed to write disclaimer log:", e)


def log_wedding_sms(result, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    for num in result.get("sent_numbers", []):
        lines.append(f"{timestamp} | SENT | {num} | {message}")

    for num in result.get("skipped_numbers", []):
        lines.append(f"{timestamp} | SKIPPED | {num} | {message}")

    for num in result.get("failed_numbers", []):
        lines.append(f"{timestamp} | FAILED | {num} | {message}")

    with open("wedding_sms_log.txt", "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_name_for_lookup(name: str) -> str:
    """
    Normalize a client name for accent-insensitive lookup.
    - Keeps original spelling intact in DB/UI
    - Strips diacritics for search keys
    - Lowercases and removes spaces
    """
    if not name:
        return ""
    # Normalize to NFKD form (decompose accents)
    nfkd = unicodedata.normalize("NFKD", name)
    # Strip combining marks (accents/diacritics)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase and remove spaces for consistent lookup
    return stripped.lower().replace(" ", "")


def log_admin_action(action, user="system"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_date = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_ROOT, f"admin_actions_{log_date}.log")



    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] by {user} | {action}\n")
    except Exception as e:
        print(f"Admin log write failed: {e}")


def find_matching_clients(name):
    cleaned = name.strip().lower()
    return Clients.query.filter(
        db.func.lower(Clients.name) == cleaned
    ).all()

def generate_unique_client_name(base_name):
    cleaned = base_name.strip()

    existing = Clients.query.filter(
        Clients.name.ilike(f"{cleaned}%")
    ).all()

    if not existing:
        return cleaned

    suffix = 2
    while True:
        candidate = f"{cleaned} ({suffix})"
        if not any(c.name == candidate for c in existing):
            return candidate
        suffix += 1

def clean_name(s):
    if not s:
        return ""
    # Collapse multiple spaces, strip ends, title case
    return " ".join(s.strip().split()).title()

def clean_mobile(s):
    if not s:
        return ""
    return s.replace(" ", "").strip()

def extract_number(value):
    """
    Extracts digits from a string like '63kg' or '178 cm' and returns an int.
    Returns None if no digits found.
    """
    if not value:
        return None
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None

def normalise_full_name(name: str) -> str:
    if not name:
        return ""

    import unicodedata
    import re

    # Strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Replace weird spaces
    name = name.replace("\xa0", " ")

    # Trim + collapse spaces
    name = " ".join(name.strip().split())

    # Split into parts (handles multi‑word names)
    parts = name.split()

    def fix_part(part):
        # Handle O', D', L', etc.
        if re.match(r"^[A-Za-z]'", part):
            return part[0].upper() + "'" + part[2:].capitalize()

        # Handle Mc / Mac prefixes
        if part.lower().startswith("mc") and len(part) > 2:
            return "Mc" + part[2:].capitalize()

        if part.lower().startswith("mac") and len(part) > 3:
            return "Mac" + part[3:].capitalize()

        # Handle hyphenated names (Smith-Jones)
        if "-" in part:
            return "-".join(p.capitalize() for p in part.split("-"))

        # Default proper case
        return part.capitalize()

    # Apply rules to each part
    cleaned = " ".join(fix_part(p) for p in parts)

    return cleaned

def levenshtein(a, b):
    # Simple pure‑Python Levenshtein distance
    if a == b:
        return 0
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]













def extract_time_window(time_frame):
    if not time_frame:
        return ""

    # Split on hyphens
    parts = [p.strip() for p in time_frame.split("-")]

    # Filter only valid HH:MM patterns
    times = [p for p in parts if len(p) == 5 and ":" in p]

    if len(times) < 2:
        return time_frame  # fallback: return original

    start = times[0]
    end = times[-1]

    return f"{start} - {end}"



# ------------------------------
# DISCLAIMER FORM FIELD MAP
# ------------------------------
HEIGHT_FIELDS = ["8", "14", "21", "28", "35", "39"]
WEIGHT_FIELDS = ["9", "15", "22", "29", "37", "38"]
NOTES_FIELDS  = ["11", "17", "25", "32", "60", "61"]

# ------------------------------
# INVITE FORM FIELD MAP
# ------------------------------
INVITE_FULLNAME_FIELDS = ["43", "46", "49", "52", "55", "58", "61", "64", "67", "70"]
INVITE_HEIGHT_FIELDS   = ["44", "47", "50", "53", "56", "59", "62", "65", "68", "71"]
INVITE_WEIGHT_FIELDS   = ["45", "48", "51", "54", "57", "60", "63", "66", "69", "72"]
INVITE_NOTES_FIELDS    = []   # none in this form


def parse_jotform_payload(payload, forced_submission_id=None):
    import json
    import unicodedata
    from sqlalchemy.util._collections import immutabledict

    # Ensure payload is a real dict
    if isinstance(payload, immutabledict):
        payload = dict(payload)

    if not isinstance(payload, dict):
        try:
            payload = json.loads(payload)
        except Exception:
            print("ERROR: Could not decode payload:", type(payload), payload)
            return []
    # 🔥 ADD DEBUG HERE — RIGHT AFTER PAYLOAD IS A DICT
    print("RAW PAYLOAD:", json.dumps(payload, indent=2))



    if forced_submission_id:
        submission_id = str(forced_submission_id)
    else:
        submission_id = str(
            payload.get("id") or
            payload.get("submission_id") or
            payload.get("form_id") or
            ""
        )



    answers = payload.get("answers", {}) or {}

    # ---------------------------------------------------------
    # AUTO-DETECT EMAIL FIELD
    # ---------------------------------------------------------
    email = ""

    # 1. Look for any control_email field
    for key, item in answers.items():
        if item.get("type") == "control_email":
            ans = item.get("answer")
            if isinstance(ans, dict):
                email = (
                    ans.get("value")
                    or ans.get("text")
                    or ans.get("full")
                    or ""
                )
            else:
                email = ans or ""
            break

    # 2. Fallback: look for fields literally named "email"
    if not email:
        for key, item in answers.items():
            if item.get("name", "").lower() == "email":
                ans = item.get("answer")
                if isinstance(ans, dict):
                    email = (
                        ans.get("value")
                        or ans.get("text")
                        or ans.get("full")
                        or ""
                    )
                else:
                    email = ans or ""
                break

    # 3. Final fallback
    email = email or ""


    # Detect which form this submission belongs to
    form_id = str(payload.get("form_id") or payload.get("id") or "")
    is_invite_form = form_id == "253599154628066"

    riders = []


    # ---------------------------------------------------------
    # EXTRACT INVITE TOKEN (supports both invite_token + i_t)
    # ---------------------------------------------------------
    invite_token = None

    # 1. Direct field #3 (invite form hidden token)
    field3 = answers.get("3")
    if field3:
        ans = field3.get("answer")
        if isinstance(ans, dict):
            invite_token = (
                ans.get("text")
                or ans.get("value")
                or ans.get("full")
            )
        else:
            invite_token = ans

    # 2. Lookup by name="i_t"
    if not invite_token:
        for f in answers.values():
            if f.get("name") == "i_t":
                ans = f.get("answer")
                if isinstance(ans, dict):
                    invite_token = (
                        ans.get("text")
                        or ans.get("value")
                        or ans.get("full")
                    )
                else:
                    invite_token = ans
                break

    # 2b. Direct lookup by key "i_t"
    if not invite_token:
        direct = answers.get("i_t")
        if direct:
            ans = direct.get("answer")
            if isinstance(ans, dict):
                invite_token = (
                    ans.get("text")
                    or ans.get("value")
                    or ans.get("full")
                )
            else:
                invite_token = ans

    # 3. Final fallback
    invite_token = invite_token or ""

    # ---------------------------------------------------------
    # DETECT AGE FIELDS (dynamic)
    # ---------------------------------------------------------
    age_fields = [
        key for key, item in answers.items()
        if item.get("type") in ("control_number", "control_dropdown")
        and "age" in item.get("text", "").lower()
    ]
    age_fields = sorted(age_fields, key=lambda x: int(x))

    # Global fields (these are disclaimer-form centric; invite form may not use them)
    guardian = answers.get("86", {}).get("answer", "") or ""
    mobile = answers.get("87", {}).get("answer", {}).get("full", "") or ""
    email = answers.get("47", {}).get("answer", "") or ""
    disclaimer = answers.get("63", {}).get("answer", None)

    def normalize_name(s):
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.replace("\xa0", " ")
        return " ".join(s.strip().lower().split())

    # ---------------------------------------------------------
    # PRELOAD CLIENTS ONCE (CRITICAL FOR PERFORMANCE)
    # ---------------------------------------------------------
    existing_clients = db.session.query(Client).all()

    client_cache = [
        (c, normalize_name(c.full_name), c.jotform_submission_id)
        for c in existing_clients
        if c.full_name
    ]

    # Exact match dictionary: norm_name -> Client
    exact_lookup = {norm: c for c, norm, _ in client_cache}

    # ---------------------------------------------------------
    # DETECT ONLY RIDER FULLNAME FIELDS
    # ---------------------------------------------------------
    if is_invite_form:
        fullname_fields = INVITE_FULLNAME_FIELDS
    else:
        # Disclaimer form uses dynamic detection
        fullname_fields = [
            key for key, item in answers.items()
            if item.get("type") == "control_fullname"
            and item.get("text", "").lower().startswith("rider")
        ]
        fullname_fields = sorted(fullname_fields, key=lambda x: int(x))

    # ---------------------------------------------------------
    # PROCESS EACH RIDER
    # ---------------------------------------------------------
    for idx, fullname_key in enumerate(fullname_fields):

        # Get the fullname field for this rider
        item = answers.get(fullname_key)
        if not item:
            continue

        # Extract rider name (prefer prettyFormat)
        pretty = item.get("prettyFormat")
        if pretty:
            raw_name = pretty
        else:
            first = (item.get("answer", {}) or {}).get("first", "")
            last = (item.get("answer", {}) or {}).get("last", "")
            raw_name = f"{first} {last}"

        name = normalize_name(raw_name)
        if not name:
            continue

        # Extract age (matched by index)
        age_key = age_fields[idx] if idx < len(age_fields) else None
        age = answers.get(age_key, {}).get("answer") if age_key else None

        # Build rider object
        rider = {
            "name": name,
            "age": age,
            "guardian": guardian,
            "mobile": mobile,
            "email": email,
            "disclaimer": disclaimer,
            "matches": [],
            "jotform_submission_id": submission_id,
            "invite_token": invite_token or ""
        }

        # ---------------------------------------------------------
        # HEIGHT / WEIGHT / NOTES (invite vs disclaimer)
        # ---------------------------------------------------------
        if is_invite_form:
            # Invite form supports up to 10 riders
            if idx < len(INVITE_HEIGHT_FIELDS):
                height_field = INVITE_HEIGHT_FIELDS[idx]
                weight_field = INVITE_WEIGHT_FIELDS[idx]
                notes_field  = INVITE_NOTES_FIELDS[idx] if idx < len(INVITE_NOTES_FIELDS) else None
            else:
                height_field = weight_field = notes_field = None
        else:
            # Disclaimer form supports up to 6 riders
            if idx < len(HEIGHT_FIELDS):
                height_field = HEIGHT_FIELDS[idx]
                weight_field = WEIGHT_FIELDS[idx]
                notes_field  = NOTES_FIELDS[idx]
            else:
                height_field = weight_field = notes_field = None

        # Extract height/weight/notes values
        height_val = answers.get(height_field, {}).get("answer") if height_field else None
        weight_val = answers.get(weight_field, {}).get("answer") if weight_field else None
        notes_val  = answers.get(notes_field, {}).get("answer") if notes_field else None

        rider["height_cm"] = extract_number(height_val)
        rider["weight_kg"] = extract_number(weight_val)
        rider["notes"] = notes_val or ""

        # ---------------------------------------------------------
        # OPTIMISED MATCHING LOGIC (FAST FOR 30,000 CLIENTS)
        # ---------------------------------------------------------
        norm_name = normalize_name(name)

        # 1. Exact match (O(1))
        matched_client = exact_lookup.get(norm_name)
        if matched_client:
            guardian_norm = normalize_name(guardian)

            # Skip guardian match
            if guardian_norm and normalize_name(matched_client.full_name) == guardian_norm:
                pass
            # Skip if already linked to this submission
            elif matched_client.jotform_submission_id != submission_id:
                rider["matches"].append(matched_client)
        else:
            # 2. Fuzzy match (only compare names starting with same letter)
            first_letter = norm_name[0] if norm_name else ""
            guardian_norm = normalize_name(guardian)

            for c, existing_norm, existing_sub_id in client_cache:
                if not existing_norm:
                    continue

                # Skip guardian
                if guardian_norm and existing_norm == guardian_norm:
                    continue

                # Skip if already linked to this submission
                if existing_sub_id == submission_id:
                    continue

                # First-letter filter (cuts most comparisons)
                if existing_norm[0] != first_letter:
                    continue

                # Fuzzy match (Levenshtein)
                if levenshtein(existing_norm, norm_name) <= 2:
                    rider["matches"].append(c)

        riders.append(rider)

    return riders



def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "")
    app.config.from_object(Config)
    app.config['SQLALCHEMY_ECHO'] = True
    db.init_app(app)

    print("IncomingSubmission columns:", IncomingSubmission.__table__.columns.keys())


    @app.route("/health")
    def health():
        return "OK", 200


    # ---------------------------------------------------------
    # HELPERS (ALL INSIDE create_app)
    # ---------------------------------------------------------

    def send_sms_clicksend(to_number, message, sender_number):
        configuration = clicksend_client.Configuration()
        configuration.username = app.config['CLICKSEND_USERNAME']
        configuration.password = app.config['CLICKSEND_API_KEY']

        api_instance = clicksend_client.SMSApi(
            clicksend_client.ApiClient(configuration)
        )

        sms_message = SmsMessage(
            source="python",
            body=message,
            to=to_number,
            _from=sender_number
        )

        sms_messages = clicksend_client.SmsMessageCollection(
            messages=[sms_message]
        )

        try:
            api_instance.sms_send_post(sms_messages)
            return True
        except ApiException as e:
            print("ClickSend SMS failed:", e)
            return False

        sms_messages = clicksend_client.SmsMessageCollection(
            messages=[sms_message]
        )

        try:
            api_instance.sms_send_post(sms_messages)
            return True
        except ApiException as e:
            print("ClickSend SMS failed:", e)
            return False

    def load_wedding_unsubscribes():
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "unsubscribe", "wedding_unsubscribe.txt")

        if not os.path.exists(path):
            return set()

        with open(path, "r") as f:
            return {line.strip() for line in f if line.strip()}

    def generate_invite_token(lesson_date_str, lesson_time_str):
        date_comp = (lesson_date_str or "").replace("-", "")
        time_comp = (lesson_time_str or "").replace(":", "")[:4]
        rand = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        return f"TR-{date_comp}-{time_comp}-{rand}"

    def teacher_times_map():
        rows = get_static_teacher_time()   # <— cached rows instead of DB query

        out = {}
        for r in rows:
            teacher_key = r.teacher_key
            weekday = str(r.weekday)
            time_val = r.time

            if teacher_key not in out:
                out[teacher_key] = {}
            if weekday not in out[teacher_key]:
                out[teacher_key][weekday] = []

            out[teacher_key][weekday].append(time_val)

        return out

    def match_submission_to_invite(sub):
        payload = json.loads(sub.raw_payload)
        riders = parse_jotform_payload(payload, forced_submission_id=sub.id)
        if not riders:
            return None

        token = (
            riders[0].get("invite_token") or
            riders[0].get("i_t") or
            riders[0].get("token")
        )

        if not token:
            return None

        invite = db.session.query(LessonInvite).filter(
                LessonInvite.token == token,
                LessonInvite.status != "completed"
        ).first()

        return invite

    def get_or_create_client_from_rider(rider, invite_mobile, submission_id):
        name = normalise_full_name(rider["name"])

        # Match by NAME ONLY
        client = db.session.query(Client).filter(
            func.lower(Client.full_name) == name.lower()
        ).first()

        if client:
            return client

        # Create new client
        client = Client(
            full_name=name,
            mobile=invite_mobile or "",
            email_primary=rider.get("email") or "",
            guardian_name=rider.get("guardian") or "",
            jotform_submission_id=submission_id
        )

        db.session.add(client)
        return client


    def create_lesson_from_invite(invite, primary_client):
        clean_time = extract_time_window(invite.time_frame)
        start, end = clean_time.split(" - ")

        lesson = Lesson(
            client_id=primary_client.client_id,
            lesson_date=invite.lesson_date,
            start_time=start,
            end_time=end,
            lesson_type=invite.lesson_type,
            group_priv=invite.group_priv,
            attendance="Pending",
            payment=None
        )

        db.session.add(lesson)
        db.session.commit()
        return lesson

    def attach_rider_to_lesson(lesson, invite, client, rider):
        # Attach rider-specific notes to the existing lesson
        note = rider.get("notes")
        if note:
            existing = lesson.notes or ""
            lesson.notes = existing + f"\n{client.full_name}: {note}"


    # Utilities
    def normalize_name_for_lookup(name):
        return re.sub(r'[^a-z]', '', (name or '').lower())

    def norm(s):
        return re.sub(r'\s+', '', (s or '').strip().lower())

    def norm_timerange_key(timerange):
        return re.sub(r'[\s:-]+', '', (timerange or '').strip())

    def to_proper_case(s):
        return ' '.join(word.capitalize() for word in (s or '').split())

    def parse_money(v):
        if v is None:
            return None
        s = str(v).strip()
        if s == '':
            return None
        s = re.sub(r'[\$,]', '', s)
        try:
            return float(s)
        except Exception:
            return None

    def parse_selected_date():
        raw = request.values.get('date') or request.values.get('selected_date') or ''
        try:
            return datetime.strptime(raw, '%Y-%m-%d').date(), raw
        except Exception:
            return None, raw

    # Register helpers with Jinja
    app.jinja_env.globals.update(
        parse_money=parse_money,
        to_proper_case=to_proper_case,
        norm=norm,
        norm_timerange_key=norm_timerange_key,
        normalize_name_for_lookup=normalize_name_for_lookup
    )

    def finalize_invite_and_submission(invite, sub, lesson):
        invite.lesson_id = lesson.lesson_id
        invite.status = "completed"
        sub.processed = True
        db.session.commit()

    @app.post("/update_lesson_field")
    def update_lesson_field():
        data = request.get_json()
        lesson_id = data.get("lesson_id")
        field = data.get("field")
        value = data.get("value")

        allowed = {"attendance", "carry_fwd", "payment", "price_pl", "adjust"}
        if field not in allowed:
            return jsonify(success=False, error="Invalid field")

        # FIXED: use the correct model name
        lesson = Lesson.query.get(lesson_id)
        if not lesson:
            return jsonify(success=False, error=f"Lesson {lesson_id} not found")

        numeric_fields = {"carry_fwd", "payment", "price_pl", "adjust"}
        if field in numeric_fields:
            try:
                value = float(value) if value not in (None, "", "None") else 0
            except:
                return jsonify(success=False, error="Invalid number")

        setattr(lesson, field, value)
        db.session.commit()

        return jsonify(success=True)


    # ---------------------------------------------------------
    # ROUTES (ALL INSIDE create_app)
    # ---------------------------------------------------------

    @app.post("/update_client_field")
    def update_client_field():
        data = request.get_json()
        client_id = data.get("client_id")
        field = data.get("field")
        value = data.get("value")

        # Allowed editable fields (EXCLUDES client name + disclaimer)
        allowed = {
            "guardian_name",
            "mobile",
            "age",
            "weight_kg",
            "height_cm",
            "email_primary",
            "email_secondary",
            "guardian2_name",
            "mobile2",
            "email_guardian2",
            "ndis_number",
            "ndis_code",
            "notes",
            "notes2"
        }

        if field not in allowed:
            return jsonify(success=False, error="Invalid field")

        client = Client.query.get(client_id)
        if not client:
            return jsonify(success=False, error="Client not found")

        # Convert numeric fields safely
        numeric_fields = {"age", "weight_kg", "height_cm"}
        if field in numeric_fields:
            try:
                value = int(value) if value else None
            except:
                return jsonify(success=False, error="Invalid number")

        setattr(client, field, value)
        db.session.commit()

        return jsonify(success=True)





    @app.route('/notifications/process/<int:sub_id>')
    def process_submission_route(sub_id):
        sub = db.session.query(IncomingSubmission).get_or_404(sub_id)
        result = process_submission(sub)
        flash(result, "success")
        return redirect(url_for('notifications'))



    @app.route('/invite_submissions')
    def invite_submissions():
        INVITE_FORM_ID = "253599154628066"

        subs = (
            db.session.query(IncomingSubmission)
            .filter(IncomingSubmission.form_id == INVITE_FORM_ID)
            .order_by(IncomingSubmission.received_at.desc())
            .all()
        )

        return render_template('invite_submissions.html', subs=subs)

    @app.route('/pending_lessons')
    def pending_lessons():
        # Get all invites that are not completed
        invites = (
            db.session.query(LessonInvite)
            .filter(LessonInvite.status != "completed")
            .order_by(LessonInvite.id.desc())
            .all()
        )

        # Get all unprocessed submissions
        submissions = (
            db.session.query(IncomingSubmission)
            .filter_by(processed=False)
            .all()
        )

        # Build lookup: invite_token → submission
        submission_lookup = {}
        for sub in submissions:
            try:
                riders = parse_jotform_payload(sub.raw_payload, forced_submission_id=sub.id)
                for rider in riders:
                    token = rider.get("invite_token")
                    if token:
                        submission_lookup[token] = sub
            except Exception:
                continue

        # Build pending list for template
        pending = []
        for inv in invites:
            sub = submission_lookup.get(inv.token)
            pending.append({
                "invite": inv,
                "submission": sub
            })

        return render_template("pending_lessons.html", pending=pending)

    @app.route("/equestrian_sms")
    def equestrian_sms():
        return render_template("equestrian_sms.html")

    @app.route("/equestrian/sms/send", methods=["POST"])
    def equestrian_sms_send():
        data = request.get_json()

        message = data.get("message", "")
        numbers = data.get("numbers", [])

        sender = app.config.get("EQUESTRIAN_SENDER", None)

        # No unsubscribe list for equestrian (yet)
        filtered = numbers[:]   # copy all numbers
        skipped = []

        sent_numbers = []
        failed_numbers = []

        for num in filtered:
            ok = send_sms_clicksend(num, message, sender)
            if ok:
                sent_numbers.append(num)
            else:
                failed_numbers.append(num)

        return jsonify({
            "ok": True,
            "sent": len(sent_numbers),
            "skipped": len(skipped),
            "failed": len(failed_numbers),
            "sent_numbers": sent_numbers,
            "skipped_numbers": skipped,
            "failed_numbers": failed_numbers,
            "status": (
                f"Sent {len(sent_numbers)} messages. "
                f"Failed {len(failed_numbers)}."
            )
        })


    @app.route("/equestrian_bulk_sms")
    def equestrian_bulk_sms():
        return render_template("equestrian_bulk_sms.html")

    @app.route("/equestrian/bulk_sms/send", methods=["POST"])
    def equestrian_bulk_sms_send():
        data = request.get_json()

        template = data.get("template", "").strip()
        rows = data.get("rows", [])
        fields = data.get("fields", [])

        sender = app.config.get("EQUESTRIAN_SENDER")

        sent_numbers = []
        failed_numbers = []
        skipped_numbers = []

        if not template:
            return jsonify({
                "ok": False,
                "status": "Template is empty.",
                "sent": 0,
                "failed": 0,
                "skipped": 0,
                "sent_numbers": [],
                "failed_numbers": [],
                "skipped_numbers": []
            })

        if not rows:
            return jsonify({
                "ok": False,
                "status": "No rows received from client.",
                "sent": 0,
                "failed": 0,
                "skipped": 0,
                "sent_numbers": [],
                "failed_numbers": [],
                "skipped_numbers": []
            })

        if "mobile" not in fields:
            return jsonify({
                "ok": False,
                "status": "CSV does not contain 'mobile' column.",
                "sent": 0,
                "failed": 0,
                "skipped": len(rows),
                "sent_numbers": [],
                "failed_numbers": [],
                "skipped_numbers": [r.get("mobile", "") for r in rows]
            })

        for row in rows:
            mobile = (row.get("mobile") or "").strip()
            if not mobile:
                skipped_numbers.append(mobile)
                continue

            personalised = template
            for field in fields:
                placeholder = f"{{{field}}}"
                if placeholder in personalised:
                    personalised = personalised.replace(placeholder, row.get(field, "") or "")

            ok = send_sms_clicksend(mobile, personalised, sender)
            if ok:
                sent_numbers.append(mobile)
            else:
                failed_numbers.append(mobile)

        status = (
            f"Sent {len(sent_numbers)} messages. "
            f"Failed {len(failed_numbers)}. "
            f"Skipped {len(skipped_numbers)}."
        )

        return jsonify({
            "ok": True,
            "status": status,
            "sent": len(sent_numbers),
            "failed": len(failed_numbers),
            "skipped": len(skipped_numbers),
            "sent_numbers": sent_numbers,
            "failed_numbers": failed_numbers,
            "skipped_numbers": skipped_numbers
        })

    def archive_wedding_sms_numbers(numbers):
        from datetime import datetime

        pending_path = os.path.join(WEDDING_SMS_ROOT, "wedding_sms_pending.txt")
        archive_path = os.path.join(
            WEDDING_SMS_ROOT,
            f"wedding_sms_sent_{datetime.now().strftime('%Y-%m-%d')}.txt"
        )


        # Append all numbers to archive
        with open(archive_path, "a", encoding="utf-8") as f:
            for num in numbers:
                f.write(num + "\n")

        # Clear pending file
        with open(pending_path, "w", encoding="utf-8") as f:
            f.write("")

    @app.get("/wedding/sms/pending")
    def wedding_sms_pending():
        path = os.path.join(WEDDING_SMS_ROOT, "wedding_sms_pending.txt")


        try:
            with open(path, "r", encoding="utf-8") as f:
                numbers = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            numbers = []

        return {"numbers": numbers}


    @app.route('/wedding/sms/send', methods=['POST'])
    def wedding_sms_send():
        data = request.get_json()

        message = data.get("message", "")
        numbers = data.get("numbers", [])

        sender = app.config['WEDDING_SENDER']

        # Load unsubscribe list
        base = os.path.dirname(os.path.abspath(__file__))
        unsub_path = os.path.join(base, "unsubscribe", "wedding_unsubscribe.txt")

        unsub = set()
        if os.path.exists(unsub_path):
            with open(unsub_path, "r") as f:
                unsub = {line.strip() for line in f if line.strip()}

        # Filter numbers
        filtered = []
        skipped = []

        for num in numbers:
            if num in unsub:
                skipped.append(num)
            else:
                filtered.append(num)

        # Send SMS to filtered numbers
        sent_numbers = []
        failed_numbers = []

        for num in filtered:
            ok = send_sms_clicksend(num, message, sender)
            if ok:
                sent_numbers.append(num)
            else:
                failed_numbers.append(num)

        # ---------------------------------------------------------
        # ARCHIVE ALL NUMBERS (Option A)
        # ---------------------------------------------------------
        archive_wedding_sms_numbers(numbers)

        return jsonify({
            "ok": True,
            "sent": len(sent_numbers),
            "skipped": len(skipped),
            "failed": len(failed_numbers),
            "sent_numbers": sent_numbers,
            "skipped_numbers": skipped,
            "failed_numbers": failed_numbers,
            "status": (
                f"Sent {len(sent_numbers)} messages. "
                f"Skipped {len(skipped)} unsubscribed numbers. "
                f"Failed {len(failed_numbers)}."
            )
        })


    @app.route('/api/teacher_times.json')
    def api_all_teacher_times():
        return {"teacher_times": teacher_times_map()}



    @app.route('/')
    def index():
        return render_template('index.html')


    @app.route('/lessons_by_date', methods=['GET', 'POST'])
    def lessons_by_date():
        selected_date, selected_date_str = parse_selected_date()
        if not selected_date:
            # fallback to today if parsing failed
            selected_date = date.today()
            selected_date_str = selected_date.strftime('%Y-%m-%d')

        # DEBUG: trace incoming date and DB sample for forensic inspection
        print("=== DEBUG lessons_by_date ENTRY ===")
        print("method:", request.method)
        print("request.values:", dict(request.values))
        raw = request.values.get('date') or request.values.get('selected_date') or ''
        print("raw date (string):", repr(raw))
        try:
            parsed = datetime.strptime(raw, '%Y-%m-%d').date() if raw else None
            print("parsed date:", parsed, "type:", type(parsed))
        except Exception as e:
            parsed = None
            print("parse error:", e)
        # quick DB check for that parsed date
        rows = db.session.query(Lesson).filter(Lesson.lesson_date == parsed).limit(5).all() if parsed else []
        print("rows found (count):", len(rows))
        for r in rows:
            print("  sample:", r.lesson_id, getattr(r, 'client', None), getattr(r, 'lesson_date', None))
        print("=== END DEBUG ===")

        grouped_lessons = {}
        horse_list = []
        horse_schedule = defaultdict(list)
        client_horse_history = defaultdict(list)



        teacher_names = [t.teacher for t in get_static_teachers()]
        block_tag_lookup = {}
        invoice_clients = []  # ✅ ensure it’s always defined

        if selected_date:
            lesson_rows = db.session.query(Lesson).filter_by(
                lesson_date=selected_date
            ).order_by(Lesson.time_frame).all()

            # Recalculate balance using carry_fwd + payment - price
            for lesson in lesson_rows:
                carry = lesson.carry_fwd or 0
                payment = lesson.payment or 0
                price = lesson.price_pl or 0
                att = (lesson.attendance or '').strip().upper()

                balance = carry + payment
                if att in ['Y', 'N']:
                    balance -= price

                lesson.balance = balance

            for lesson in lesson_rows:
                horse_name = to_proper_case(lesson.horse)
                att = (lesson.attendance or '').strip().upper()
                time_frame = (lesson.time_frame or '').strip()

                if horse_name and time_frame and att != 'C':
                    horse_schedule[horse_name].append(time_frame)

            time_lookup = {norm(t.timerange): t for t in db.session.query(Time).all()}
            client_lookup = {}
            for c in db.session.query(Client).all():
                full = c.full_name or ''
                key = normalize_name_for_lookup(full)
                client_lookup[key] = c
                client_lookup[(full or '').lower().replace(' ', '')] = c

            block_tag_lookup = {}
            rows = db.session.query(LessonBlockTag).filter_by(lesson_date=selected_date).all()
            for r in rows:
                key = norm_timerange_key(r.time_range)
                tags = [t.strip() for t in (r.teacher_tags or '').split(',') if t.strip()]
                block_tag_lookup[key] = tags

            grouped = defaultdict(list)
            for lesson in lesson_rows:
                time_key = norm(lesson.time_frame)
                time_obj = time_lookup.get(time_key)
                client_key = normalize_name_for_lookup(lesson.client or '')
                client_obj = client_lookup.get(client_key)

                if not client_obj:
                    alt_key = (lesson.client or '').lower().replace(' ', '')
                    client_obj = client_lookup.get(alt_key)

                timerange_display = time_obj.timerange if time_obj else (lesson.time_frame or '—')
                group_key = (timerange_display, lesson.lesson_type or '', lesson.group_priv or '')
                grouped[group_key].append((lesson, client_obj))

            def time_sort_key(timerange):
                try:
                    return datetime.strptime(timerange.split('-')[0].strip(), '%H:%M').time()
                except Exception:
                    return time.min   # earliest possible time of day

            sorted_keys = sorted(
                grouped.keys(),
                key=lambda k: (0 if (k[1] or '').lower() == 'arena' else 1, time_sort_key(k[0]))
            )

            grouped_lessons = {
                k: sorted(grouped[k], key=lambda pair: ((pair[1].full_name if pair[1] else pair[0].client) or '').lower())
                for k in sorted_keys
            }

            # ✅ Seed defaults AFTER grouped_lessons is built
            for (timerange, lesson_type, group_priv), lesson_group in grouped_lessons.items():
                block_key = norm_timerange_key(timerange)
                saved_tags = block_tag_lookup.get(block_key)
                if not saved_tags or saved_tags == ['']:
                    if (lesson_type or '').lower() == 'arena':
                        block_tag_lookup[block_key] = ['T1']
                    else:
                        block_tag_lookup[block_key] = ['T2']

            # Debug print to confirm
            print("block_tag_lookup (final):", block_tag_lookup)

            horse_list = [to_proper_case(h.horse) for h in get_static_horses()]
            horse_schedule = {str(k): [str(t) for t in v] for k, v in horse_schedule.items()}

            invoice_clients = [
                c.client_id for c in db.session.query(Client).filter_by(invoice_required=True).all()
            ]

            # Populate clients for the popup client dropdown
            clients = get_static_clients()
            weekday_int = selected_date.weekday()            # Python’s 0=Monday … 6=Sunday
            print("teacher_times_map:", teacher_times_map())
            print("weekday_int passed:", weekday_int)

            # ✅ Query all timeranges for Finish dropdowns in popups
            time_rows = get_static_times()
            times = [row.timerange for row in time_rows]
            
            teacher_horse_usage = {}
            selected_date_obj = selected_date


            teacher_horse_usage = {}
            # selected_date is already a datetime.date, so use it directly
            selected_date_obj = selected_date

            teacher_horse_usage = {}
            # selected_date is already a datetime.date, so use it directly
            selected_date_obj = selected_date

            teacher_rows = db.session.query(TeacherHorse).filter(
                        TeacherHorse.date == selected_date_obj
            ).all()

            for th in teacher_rows:
                        bk = (th.block_key or "").strip()
                        if len(bk) == 8 and bk.isdigit():
                                    start, end = bk[:4], bk[4:]
                                    time_slot = f"{start[:2]}:{start[2:]} - {end[:2]}:{end[2:]}"
                        else:
                                    time_slot = bk or "??:??"

                        for raw_h in (th.horse1, th.horse2):
                                    if raw_h:
                                                name = to_proper_case(str(raw_h).strip())
                                                teacher_horse_usage.setdefault(name, []).append(time_slot)

            for h, times in teacher_horse_usage.items():
                        teacher_horse_usage[h] = sorted(set(times))


        return render_template(
            'lessons_by_date.html',
            grouped_lessons=grouped_lessons,
            selected_date=selected_date_str,
            weekday_int=weekday_int,
            horse_list=horse_list,
            horse_schedule=horse_schedule,
            teacher_names=teacher_names,
            block_tag_lookup=block_tag_lookup,
            client_horse_history=client_horse_history,
            invoice_clients=invoice_clients,  # ✅ Add this line
            clients=clients, 
            teacher_times=teacher_times_map(),   # 👈 add this
            teacher_horse_usage=teacher_horse_usage,   # 👈 add this
            times=times   # 👈 add this

        )




    def detect_conflicts(riders):
        conflicts = []
        for rider in riders:
            if rider.get("matches"):
                conflicts.append({
                    "rider": rider,
                    "matches": rider["matches"]
                })
        return conflicts


    def create_lesson_row(invite, client_name):
        """
        Creates one Lesson row for a single client using fields from lesson_invites.
        """
        clean_time = extract_time_window(invite.time_frame)

        lesson = Lesson(
            lesson_date=invite.lesson_date,
            time_frame=clean_time,
            lesson_type=invite.lesson_type,
            group_priv=invite.group_priv,
            price_pl=invite.cost_per_person,
            client=client_name,
            horse=""
        )

        db.session.add(lesson)
        db.session.flush()

        return lesson



    def process_submission(sub):
        """
        Robust replacement for the old process_submission.
        """

        # ---------------------------------------------------------
        # TESTING MODE (set True while developing)
        # ---------------------------------------------------------
        TESTING_MODE = True

        # ---------------------------------------------------------
        # MATCH INVITE FIRST (we need invite.status for guardrail)
        # ---------------------------------------------------------
        invite = match_submission_to_invite(sub)
        if not invite:
            return "No matching invite"

        submission_id = str(sub.id)


        # ---------------------------------------------------------
        # PRODUCTION GUARDRAIL:
        # Only skip if the invite is already completed.
        # ---------------------------------------------------------
        if not TESTING_MODE:
            if invite.status == "completed":
                sub.processed = True
                db.session.commit()
                return f"Invite already completed for submission {submission_id}"
        # In TESTING_MODE, we allow full reprocessing.

        # ---------------------------------------------------------
        # PARSE RIDERS
        # ---------------------------------------------------------
        try:
            riders = parse_jotform_payload(
                sub.raw_payload,
                forced_submission_id=sub.id
            )
        except Exception as e:
            print("RIDER PARSE ERROR:", e)
            return f"Invalid payload: {e}"

        if not riders:
            return "No riders found"

        # ---------------------------------------------------------
        # CONFLICT DETECTION
        # ---------------------------------------------------------
        conflicts = detect_conflicts(riders) if 'detect_conflicts' in globals() else None
        if conflicts:
            session['conflicts'] = conflicts
            session['sub_id_pending_conflict'] = sub.id
            return "conflict"

        # ---------------------------------------------------------
        # PREPARE INVITE MOBILE
        # ---------------------------------------------------------
        def convert_mobile_for_client(m):
            if not m:
                return ""
            digits = re.sub(r'\D+', '', str(m))
            if digits.startswith("61") and len(digits) > 2:
                return "0" + digits[2:]
            return digits

        invite_mobile = convert_mobile_for_client(invite.mobile)

        # ---------------------------------------------------------
        # CREATE / ATTACH LESSONS
        # ---------------------------------------------------------
        created_lessons = []
        errors = []

        clean_time = extract_time_window(invite.time_frame)

        for rider in riders:
            try:
                client = get_or_create_client_from_rider(
                    rider,
                    invite_mobile,
                    submission_id
                )

                if rider.get("height_cm") is not None:
                    client.height_cm = rider.get("height_cm")
                if rider.get("weight_kg") is not None:
                    client.weight_kg = rider.get("weight_kg")
                if rider.get("notes"):
                    client.notes = rider.get("notes")

                print("DEBUG MATCH CHECK:")
                print("  lesson_date:", invite.lesson_date, type(invite.lesson_date))
                print("  clean_time:", clean_time)
                print("  lesson_type:", invite.lesson_type)
                print("  client.full_name:", client.full_name)

                existing_lesson = db.session.query(Lesson).filter(
                    Lesson.lesson_date == invite.lesson_date,
                    Lesson.time_frame == clean_time,
                    Lesson.lesson_type == invite.lesson_type,
                    func.lower(func.trim(Lesson.client)) == func.lower(func.trim(client.full_name))
                ).first()

                if existing_lesson:
                    created_lessons.append(existing_lesson)
                    client.jotform_submission_id = submission_id
                    continue
                print("  existing_lesson:", existing_lesson)

                lesson = Lesson(
                    lesson_date=invite.lesson_date,
                    time_frame=clean_time,
                    lesson_type=invite.lesson_type,
                    group_priv=invite.group_priv,
                    price_pl=invite.cost_per_person,
                    client=(client.full_name or ""),
                    horse="",
                    attendance="Pending",
                    payment=None
                )

                db.session.add(lesson)
                db.session.flush()

                client.jotform_submission_id = submission_id
                created_lessons.append(lesson)

            except Exception as e:
                print("RIDER ERROR:", rider, e)   # 👈 add this  
                errors.append(str(e))
                continue

        # ---------------------------------------------------------
        # FINALIZE
        # ---------------------------------------------------------
        if created_lessons:
            try:
                invite.lesson_id = created_lessons[0].lesson_id
            except Exception:
                invite.lesson_id = getattr(created_lessons[0], "lesson_id", None)

        invite.status = "completed"
        sub.processed = True

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return f"DB commit failed: {e}"

        if errors:
            return f"Processed {len(created_lessons)} riders into lessons; errors: {len(errors)}"
        return f"Processed {len(created_lessons)} riders into lessons"



    # ---------------- ROUTES ---------------- #

    @app.route('/client/change_name/<int:client_id>', methods=['POST'])
    def change_client_name(client_id):
        new_name = request.form.get('new_name', '').strip()
        if not new_name:
            flash("New name cannot be empty.", "danger")
            return redirect(url_for('client_view', client=client_id))

        client = db.session.query(Client).get(client_id)
        if not client:
            flash("Client not found.", "danger")
            return redirect(url_for('client_view'))

        old_name = client.full_name

        # Update client record
        client.full_name = new_name

        # Update all lessons referencing the old name
        db.session.query(Lesson).filter(
            func.lower(func.trim(Lesson.client)) == func.lower(func.trim(old_name))
        ).update(
            {Lesson.client: new_name},
            synchronize_session=False
        )

        db.session.commit()

        flash(f"Name updated from '{old_name}' to '{new_name}'.", "success")
        return redirect(url_for('client_view', client=client_id))


    @app.route('/client_history/<client_key>')
    def client_history(client_key):
        clean = client_key.strip().lower()

        today = date.today()

        rows = (
            db.session.query(Lesson)
            .filter(
                func.replace(func.lower(Lesson.client), " ", "") == clean
            )
            .filter(Lesson.lesson_date < today)   # <-- ONLY past lessons
            .order_by(Lesson.lesson_date.desc(), Lesson.lesson_id.desc())
            .limit(10)
            .all()
        )

        horses = [(r.horse or "").strip() for r in rows]

        return jsonify(horses)

    
    @app.route('/send_invite', methods=['POST'])
    def send_invite():
        # Debug: see exactly what the popup sent
        print("SMS INVITE POST:", dict(request.form))

        # Accept both the new popup invite_* names and any old ones
        lesson_id_raw = (
            request.form.get("invite_lesson_id") or
            request.form.get("lesson_id")
        )

        lesson_date_str = (
            request.form.get("invite_date") or
            request.form.get("lesson_date") or
            request.form.get("date")
        )

        # Read posted time fields (prefer the full time_frame)
        invite_time_frame = (request.form.get("invite_time_frame") or "").strip()
        invite_lesson_type = (
                request.form.get("invite_lesson_type")
                or request.form.get("lesson_type")
                or ""
        ).strip()

        invite_group_priv = (
                request.form.get("invite_group_priv")
                or request.form.get("group_priv")
                or ""
        ).strip()

        invite_duration = (request.form.get("invite_duration") or "").strip()   # minutes as string
        invite_end = (request.form.get("invite_end") or "").strip()             # HH:MM if provided

        # Backwards-compatible single-value fields
        lesson_time_fallback = (
            request.form.get("invite_time") or
            ""
        ).strip()

        # Build canonical lesson_time_str (full "HH:MM - HH:MM") when possible
        # ALWAYS use the canonical frontend time frame
        lesson_time_str = invite_time_frame

        mobile_raw = (
            request.form.get("invite_mobile") or
            request.form.get("mobile") or
            request.form.get("client_phone") or
            ""
        )

        riders_requested_raw = (
            request.form.get("invite_riders") or
            request.form.get("riders_requested") or
            "1"
        )

        cost_per_person_raw = (
            request.form.get("invite_cost") or
            request.form.get("cost_per_person") or
            "0"
        )

        # Normalise to +61 format for ClickSend
        digits = re.sub(r'\D+', '', mobile_raw or "")
        if digits.startswith("0"):
            digits = digits[1:]
        mobile_clean = "+61" + digits if digits else ""

        try:
            riders_requested = int(riders_requested_raw)
        except Exception:
            riders_requested = 1

        try:
            cost_per_person = float(cost_per_person_raw)
        except Exception:
            cost_per_person = 0.0

        # lesson_id may be None or a non-numeric settime_... token
        lesson_id = int(lesson_id_raw) if lesson_id_raw and str(lesson_id_raw).isdigit() else None

        if not lesson_date_str or not lesson_time_str or not mobile_clean:
            flash("Missing lesson date, time, or mobile number for invite.", "danger")
            return redirect(url_for('lessons_by_date', date=lesson_date_str or date.today().isoformat()))

        # Generate token (use canonical lesson_time_str)
        token = generate_invite_token(lesson_date_str, lesson_time_str)

        # Build Jotform URL (query params)
        base_url = "https://form.jotform.com/253599154628066"
        params = urlencode({
            "i_t": token,
            "r_r": riders_requested
        })
        jotform_url = f"{base_url}?{params}"
        print("JOTFORM URL:", jotform_url)

        # Debug prints (temporary)
        print("DEBUG: invite_time_frame (posted):", invite_time_frame)
        print("DEBUG: invite_duration (posted):", invite_duration)
        print("DEBUG: invite_end (posted/computed):", invite_end)
        print("DEBUG: lesson_time_str (canonical):", lesson_time_str)

        invite = LessonInvite(
            lesson_id=lesson_id,
            token=token,
            mobile=mobile_clean,
            riders_requested=riders_requested,
            cost_per_person=cost_per_person,
            time_frame=lesson_time_str,   # ALWAYS canonical "HH:MM - HH:MM"
            lesson_type=invite_lesson_type,
            group_priv=invite_group_priv,
            status="awaiting_form",
            lesson_date=datetime.strptime(lesson_date_str, "%Y-%m-%d").date()
        )

        db.session.add(invite)
        db.session.commit()

        # Build SMS text
        total_cost = riders_requested * cost_per_person

        # Reformat date from YYYY-MM-DD → DD Mon
        from dateutil import parser

        try:
            date_obj = parser.parse(lesson_date_str)
            formatted_date = date_obj.strftime("%-d %b")   # Linux/macOS
        except Exception:
            try:
                formatted_date = date_obj.strftime("%#d %b")  # Windows fallback
            except:
                formatted_date = lesson_date_str

        # Parse the start time robustly from the canonical lesson_time_str
        m = re.search(r'(\d{1,2}:\d{2})', lesson_time_str or "")
        start_24 = m.group(1) if m else (lesson_time_str.split("-")[0].strip() if lesson_time_str else "")

        # Convert to 12-hour format with dot and am/pm → "10.00am", "2.30pm"
        try:
            t_obj = datetime.strptime(start_24, "%H:%M")
            hour_12 = t_obj.strftime("%I").lstrip("0")
            minute = t_obj.strftime("%M")
            ampm = t_obj.strftime("%p").lower()
            start_pretty = f"{hour_12}.{minute}{ampm}"
        except Exception:
            start_pretty = start_24 or ""

        # For now we keep the full JotForm URL; replace with shortener later if desired
        short_url = jotform_url

        # Plural fix
        rider_word = "rider" if int(riders_requested) == 1 else "riders"

        sms_text = (
            f"Hi! Please confirm your lesson on {formatted_date} at {start_pretty}. "
            f"{riders_requested} {rider_word}, ${cost_per_person:.0f} ea. "
            f"{short_url}"
        )

        # Debug prints to verify canonical values
        print("CANONICAL invite_time_frame:", invite.time_frame)
        print("lesson_time_str used:", lesson_time_str)
        print("parsed start_24:", start_24, "start_pretty:", start_pretty)

        ok = send_sms_clicksend(mobile_clean, sms_text)
        if ok:
            flash("SMS invite sent successfully.", "success")
        else:
            flash("Failed to send SMS invite via ClickSend.", "danger")

        print("SMS INVITE POST:", request.form)

        return redirect(url_for('lessons_by_date', date=lesson_date_str))

    @app.route("/process_invite/<int:invite_id>")
    def process_invite(invite_id):
        invite = db.session.query(LessonInvite).get(invite_id)
        if not invite:
            return "Invite not found."

        sub = None
        pending = db.session.query(IncomingSubmission).filter_by(processed=False).all()

        for s in pending:
            try:
                riders = parse_jotform_payload(s.raw_payload)
                for rider in riders:
                    token = rider.get("invite_token")
                    if token == invite.token:
                        sub = s
                        break
                if sub:
                    break
            except Exception:
                continue

        if not sub:
            return "No matching submission found."

        invite.status = "form_received"
        db.session.commit()

        process_submission(sub)

        return redirect(url_for("pending_lessons"))



    # ---------------- ROUTES ---------------- #

    @app.route('/wedding/sms', methods=['GET'])
    def wedding_sms():
        return render_template('wedding_sms.html')

    @app.route("/purge_jotform_invites")
    def purge_jotform_invites():
        import requests

        API_KEY = os.getenv("JOTFORM_API_KEY", "")
        FORM_ID = "253599154628066"

        # Fetch all submissions
        url = f"https://api.jotform.com/form/{FORM_ID}/submissions?apiKey={API_KEY}"
        data = requests.get(url).json()
        subs = data.get("content", [])

        count = 0
        for s in subs:
            sid = s.get("id")
            if sid:
                del_url = f"https://api.jotform.com/submission/{sid}?apiKey={API_KEY}"
                requests.delete(del_url)
                count += 1

        return f"Deleted {count} JotForm submissions."


    @app.route('/u/<code>')
    def short_redirect(code):
        if code in short_links:
            return redirect(short_links[code])
        return "Invalid or expired link", 404

    @app.route("/client/<int:client_id>/change_price", methods=["POST"])
    def change_client_price(client_id):
        mode = request.form.get("mode")
        group_priv = request.form.get("group_priv")
        cutoff_date = request.form.get("cutoff_date")
        new_price = request.form.get("new_price")

        # Get the actual client name from the DB
        client_obj = Client.query.get(client_id)
        client_name = client_obj.full_name

        if not group_priv:
            flash("Group Priv is required.", "danger")
            return redirect(url_for("client_view", client_id=client_id))

        if not new_price:
            flash("New price is required.", "danger")
            return redirect(url_for("client_view", client_id=client_id))

        db.session.execute(
            text("""
                UPDATE lessons
                SET price_pl = :new_price
                WHERE client = :client_name
                  AND group_priv = :group_priv
                  AND lesson_date >= :cutoff_date
            """),
            {
                "new_price": new_price,
                "client_name": client_name,
                "group_priv": group_priv,
                "cutoff_date": cutoff_date
            }
        )

        db.session.commit()
        return redirect(url_for("client_view", client_id=client_id))





    @app.route("/client/<int:client_id>/change_horse", methods=["POST"])
    def change_client_horse(client_id):
        mode = request.form.get("mode")
        group_priv = request.form.get("group_priv")
        cutoff_date = request.form.get("cutoff_date")
        new_horse_id = request.form.get("new_horse")

        # Convert horse_id → horse name
        horse_obj = Horse.query.get(new_horse_id)
        new_horse_name = horse_obj.horse

        # Get client name
        client_obj = Client.query.get(client_id)
        client_name = client_obj.full_name

        if not group_priv:
            flash("Group Priv is required.", "danger")
            return redirect(url_for("client_view", client_id=client_id))

        if mode == "change_horse":
            db.session.execute(
                text("""
                    UPDATE lessons
                    SET horse = :new_horse
                    WHERE client = :client_name
                      AND group_priv = :group_priv
                      AND lesson_date >= :cutoff_date
                """),
                {
                    "new_horse": new_horse_name,
                    "client_name": client_name,
                    "group_priv": group_priv,
                    "cutoff_date": cutoff_date
                }
            )

        elif mode == "assign_if_empty":
            db.session.execute(
                text("""
                    UPDATE lessons
                    SET horse = :new_horse
                    WHERE client = :client_name
                      AND group_priv = :group_priv
                      AND lesson_date >= :cutoff_date
                      AND (horse IS NULL OR horse = '')
                """),
                {
                    "new_horse": new_horse_name,
                    "client_name": client_name,
                    "group_priv": group_priv,
                    "cutoff_date": cutoff_date
                }
            )

        db.session.commit()
        return redirect(url_for("client_view", client_id=client_id))



    @app.route('/debug', methods=['GET', 'POST'])
    def debug_page():
        selected_client = request.args.get('client') or request.form.get('client')

        # Query all clients
        q = db.session.query(Client).order_by(Client.full_name.asc()).all()

        clients = []
        for row in q:
            if isinstance(row, (list, tuple)):
                v = row[0]
            else:
                v = getattr(row, 'full_name', None) if hasattr(row, 'full_name') else str(row)

            if v:
                v = str(v).strip()
                if v:
                    clients.append(v)

        # Deduplicate
        seen = set()
        clients = [c for c in clients if not (c in seen or seen.add(c))]

        print("DEBUG clients payload (count):", len(clients), "sample:", clients[:50])

        return render_template('debug.html', clients=clients, selected_client=selected_client)

    @app.route('/notifications')
    def notifications():
        rows = db.session.query(IncomingSubmission).filter_by(
            processed=False
        ).order_by(
            IncomingSubmission.received_at.desc()
        ).all()

        # Extract rider names using the same parser used for processing
        for r in rows:
            try:
                riders = parse_jotform_payload(r.raw_payload)
                r.display_names = ", ".join([rider["name"] for rider in riders]) if riders else "(unknown)"
            except Exception as e:
                print("ERROR extracting names:", e)
                r.display_names = "(unknown)"

        return render_template(
            'notifications.html',
            rows=rows
        )

        
    @app.route('/notifications/fetch')
    def fetch_jotform_submissions():
        import requests
        import json

        # Your hard‑coded credentials
        API_KEY = os.getenv("JOTFORM_API_KEY", "")
        FORM_ID = "211021514885045"

        url = f"https://api.jotform.com/form/{FORM_ID}/submissions?apiKey={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            print("ERROR: JotForm API failed:", response.text)
            return redirect(url_for('notifications'))

        data = response.json()
        submissions = data.get("content", [])

        inserted = 0

        for sub in submissions:
            # Extract the TRUE submission ID from the JSON
            submission_id = str(sub.get("id"))
            print("JOTFORM ID:", submission_id)

            # Skip if this submission already exists in incoming_submissions

            # Compute stable hash of the entire payload
            payload_str = json.dumps(sub, sort_keys=True)
            payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()

            # Dedupe by hash (blocks JotForm replay duplicates)
            existing_row = db.session.query(IncomingSubmission).filter_by(
                form_id=submission_id
            ).first()

            if existing_row:
                continue

            # Insert into incoming_submissions
            row = IncomingSubmission(
                form_id=submission_id,               # store REAL submission ID
                raw_payload=json.dumps(sub),         # store full JSON
                processed=False,
                unique_hash=payload_hash
            )
            db.session.add(row)
            inserted += 1

        db.session.commit()
        print(f"Inserted {inserted} new submissions.")

        return redirect(url_for('notifications'))



    @app.route('/notifications/<int:webhook_id>')
    def process_notification(webhook_id):
        submission = db.session.query(IncomingSubmission).get_or_404(webhook_id)
        riders = parse_jotform_payload(submission.raw_payload, forced_submission_id=submission.form_id)
        # Unified conflict detection (same as batch)
        for i, rider in enumerate(riders, start=1):
            if rider.get("matches"):
                return redirect(url_for(
                    'resolve_conflict',
                    submission_id=submission.id,
                    rider_index=i
                ))

        # No conflicts → show normal processing screen
        return render_template(
            'process_notification.html',
            submission=submission,
            clients=riders
        )


    @app.route('/notifications/conflict/<int:submission_id>/<int:rider_index>')
    def resolve_conflict(submission_id, rider_index):
        row = db.session.query(IncomingSubmission).get_or_404(submission_id)
        riders = parse_jotform_payload(row.raw_payload, forced_submission_id=row.form_id)
        # rider_index is 1‑based
        rider = riders[rider_index - 1]

        matches = rider.get("matches", [])

        return render_template(
            'conflict_resolution.html',
            submission=row,
            rider=rider,
            matches=matches,
            rider_index=rider_index
        )


    @app.route('/notifications/conflict/<int:submission_id>/<int:rider_index>', methods=['POST'])
    def finalize_conflict(submission_id, rider_index):
        choice = request.form.get("choice")
        client_id = request.form.get("client_id")

        row = db.session.query(IncomingSubmission).get_or_404(submission_id)
        riders = parse_jotform_payload(row.raw_payload, forced_submission_id=row.form_id)
        rider = riders[rider_index - 1]

        # Clean incoming fields
        name = normalise_full_name(rider["name"])
        age = int(rider["age"]) if rider["age"] else None
        guardian = rider["guardian"]
        mobile = clean_mobile(rider["mobile"])
        email = rider["email"]
        disclaimer = int(rider["disclaimer"]) if rider["disclaimer"] else None
        jotform_id = str(row.form_id)

        # Fetch selected existing client (if any)
        existing = Client.query.get(client_id) if client_id else None

        # -----------------------------------------
        # USE EXISTING
        # -----------------------------------------
        if choice == "use_existing" and existing:
            existing.jotform_submission_id = jotform_id
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # OVERWRITE EXISTING
        # -----------------------------------------
        if choice == "overwrite" and existing:
            existing.full_name = name
            existing.age = age
            existing.guardian_name = guardian
            existing.mobile = mobile
            existing.email_primary = email
            existing.disclaimer = disclaimer
            existing.jotform_submission_id = jotform_id
            existing.height_cm = rider.get("height_cm")
            existing.weight_kg = rider.get("weight_kg")
            existing.notes = rider.get("notes")
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # CREATE NEW CLIENT
        # -----------------------------------------
        if choice == "new":
            new_client = Client(
                full_name=name,
                age=age,
                guardian_name=guardian,
                mobile=mobile,
                email_primary=email,
                disclaimer=disclaimer,
                height_cm=rider.get("height_cm"),
                weight_kg=rider.get("weight_kg"),
                notes=rider.get("notes"),
                invoice_required=False,
                jotform_submission_id=jotform_id
            )
            db.session.add(new_client)
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # CREATE NEW CLIENT (SAME NAME)
        # -----------------------------------------
        if choice == "new_same_name":
            base = name
            counter = 2

            while True:
                candidate = f"{base} ({counter})"
                exists = db.session.query(Client).filter_by(full_name=candidate).first()
                if not exists:
                    break
                counter += 1

            new_client = Client(
                full_name=candidate,
                age=age,
                guardian_name=guardian,
                mobile=mobile,
                email_primary=email,
                disclaimer=disclaimer,
                height_cm=rider.get("height_cm"),
                weight_kg=rider.get("weight_kg"),
                notes=rider.get("notes"),
                invoice_required=False,
                jotform_submission_id=jotform_id
            )
            db.session.add(new_client)
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # FALLBACK
        # -----------------------------------------
        db.session.commit()
        return redirect(url_for('process_all_pending'))




    @app.route('/notifications/<int:webhook_id>', methods=['POST'])
    def finalize_notification(webhook_id):
        row = db.session.query(IncomingSubmission).get_or_404(webhook_id)
        payload = row.raw_payload
        riders = parse_jotform_payload(payload, forced_submission_id=row.form_id)

        # The REAL jotform submission ID you stored
        jotform_id = str(row.form_id)

        for i, rider in enumerate(riders, start=1):
            choice = request.form.get(f"client_choice_{i}")

            name = request.form.get(f"name_{i}")
            age = request.form.get(f"age_{i}")
            guardian = request.form.get(f"guardian_{i}")

            raw_mobile = request.form.get(f"mobile_{i}")
            mobile = clean_mobile(raw_mobile)

            email = request.form.get(f"email_{i}")

            raw_disclaimer = request.form.get(f"disclaimer_{i}")
            try:
                disclaimer = int(raw_disclaimer) if raw_disclaimer else None
            except ValueError:
                disclaimer = None

            # NEW FIELDS
            height_cm = extract_number(request.form.get(f"height_{i}"))
            weight_kg = extract_number(request.form.get(f"weight_{i}"))
            notes = request.form.get(f"notes_{i}") or ""

            # --- CREATE NEW ---
            if choice == "new":
                final_name = name

                new_client = Client(
                    full_name=final_name,
                    guardian_name=guardian,
                    age=age,
                    mobile=mobile,
                    email_primary=email,
                    disclaimer=disclaimer,
                    height_cm=height_cm,
                    weight_kg=weight_kg,
                    notes=notes,
                    invoice_required=False,
                    jotform_submission_id=jotform_id
                )
                db.session.add(new_client)
                db.session.flush()  # ensure client_id is available
                log_submission_link("CREATE_NEW", new_client, jotform_id)
                continue

            # --- CREATE NEW (SAME NAME) ---
            if choice == "new_same_name":
                final_name = generate_unique_client_name(name)

                new_client = Client(
                    full_name=final_name,
                    guardian_name=guardian,
                    age=age,
                    mobile=mobile,
                    email_primary=email,
                    disclaimer=disclaimer,
                    height_cm=height_cm,
                    weight_kg=weight_kg,
                    notes=notes,
                    invoice_required=False,
                    jotform_submission_id=jotform_id
                )
                db.session.add(new_client)
                db.session.flush()
                log_submission_link("CREATE_NEW_SAME_NAME", new_client, jotform_id)
                continue

            # --- USE EXISTING ---
            if choice.startswith("existing_"):
                existing_id = int(choice.replace("existing_", ""))
                client = Client.query.get(existing_id)
                if client:
                    client.jotform_submission_id = jotform_id
                    log_submission_link("USE_EXISTING", client, jotform_id)
                continue

            # --- OVERWRITE EXISTING ---
            if choice.startswith("overwrite_"):
                existing_id = int(choice.replace("overwrite_", ""))
                client = Client.query.get(existing_id)

                client.full_name = name
                client.guardian_name = guardian
                client.age = age
                client.mobile = mobile
                client.email_primary = email
                client.disclaimer = disclaimer
                client.jotform_submission_id = jotform_id

                # NEW FIELDS
                client.height_cm = height_cm
                client.weight_kg = weight_kg
                client.notes = notes

                log_submission_link("OVERWRITE_EXISTING", client, jotform_id)
                continue

        # Log disclaimer names
        riders_for_log = parse_jotform_payload(row.raw_payload, forced_submission_id=row.form_id)
        names = [r["name"] for r in riders_for_log]
        log_disclaimer_processed(names)

        # Mark submission processed
        row.processed = True
        row.processed_at = datetime.utcnow()
        db.session.commit()


        return redirect(url_for('notifications'))

    @app.route('/notifications/process_all')
    def process_all_pending():
        from datetime import datetime

        # Get the next unprocessed submission
        next_row = db.session.query(IncomingSubmission).filter_by(processed=False).first()

        if not next_row:
            return redirect(url_for('notifications'))

        # Parse riders from stored JSON
        riders = parse_jotform_payload(next_row.raw_payload, forced_submission_id=next_row.form_id)

        # ---------------------------------------------------------
        # FIX 1: CLEAR MATCHES FOR RIDERS ALREADY RESOLVED
        # ---------------------------------------------------------
        for rider in riders:
            existing = db.session.query(Client).filter_by(
                jotform_submission_id=rider["jotform_submission_id"]
            ).first()

            if existing:
                rider["matches"] = []

        # ---------------------------------------------------------
        # FIX 3: IF ALL RIDERS ARE RESOLVED → MARK PROCESSED
        # ---------------------------------------------------------
        all_resolved = True
        for rider in riders:
            existing = db.session.query(Client).filter_by(
                jotform_submission_id=rider["jotform_submission_id"]
            ).first()

            if not existing:
                all_resolved = False
                break

        if all_resolved:
            # Log disclaimer names
            riders_for_log = parse_jotform_payload(next_row.raw_payload, forced_submission_id=next_row.form_id)
            names = [r["name"] for r in riders_for_log]
            log_disclaimer_processed(names)

            next_row.processed = True
            next_row.processed_at = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # STEP 1: CHECK FOR CONFLICTS
        # -----------------------------------------
        for idx, rider in enumerate(riders, start=1):
            matches = rider.get("matches", [])
            if matches:
                return redirect(url_for(
                    'finalize_conflict',
                    submission_id=next_row.id,
                    rider_index=idx
                ))

        # -----------------------------------------
        # STEP 2: NO CONFLICTS → CREATE NEW CLIENTS
        # -----------------------------------------
        for rider in riders:
            name = normalise_full_name(rider["name"])
            age = int(rider["age"]) if rider["age"] else None
            guardian = rider["guardian"]
            mobile = clean_mobile(rider["mobile"])
            email = rider["email"]
            disclaimer = int(rider["disclaimer"]) if rider["disclaimer"] else None
            jotform_id = rider["jotform_submission_id"]

            new_client = Client(
                full_name=name,
                age=age,
                guardian_name=guardian,
                mobile=mobile,
                email_primary=email,
                disclaimer=disclaimer,
                height_cm=rider.get("height_cm"),
                weight_kg=rider.get("weight_kg"),
                notes=rider.get("notes"),
                jotform_submission_id=jotform_id
            )
            db.session.add(new_client)

        # -----------------------------------------
        # STEP 3: MARK SUBMISSION AS PROCESSED
        # -----------------------------------------

        # Log disclaimer names
        riders_for_log = parse_jotform_payload(next_row.raw_payload, forced_submission_id=next_row.form_id)
        names = [r["name"] for r in riders_for_log]
        log_disclaimer_processed(names)

        next_row.processed = True
        next_row.processed_at = datetime.utcnow()
        db.session.commit()

        return redirect(url_for('process_all_pending'))






    @app.route("/fetch_invite_submissions")
    def fetch_invite_submissions():
        import requests
        import json
        from datetime import datetime

        INVITE_FORM_ID = "253599154628066"
        API_KEY = os.getenv("JOTFORM_API_KEY", "")

        url = f"https://api.jotform.com/form/{INVITE_FORM_ID}/submissions?apiKey={API_KEY}"

        r = requests.get(url)
        data = r.json()

        submissions = data.get("content", [])

        count = 0

        for s in submissions:
            jot_id = s.get("id")

            # Skip if already stored
            existing = db.session.query(IncomingSubmission).filter_by(form_id=jot_id).first()
            if existing:
                continue

            raw_payload = json.dumps(s)

            new_sub = IncomingSubmission(
                form_id=jot_id,
                raw_payload=raw_payload,
                received_at=datetime.utcnow(),
                processed=False
            )
            db.session.add(new_sub)
            db.session.commit()

            # Attempt to match to invite immediately
            invite = match_submission_to_invite(new_sub)
            if invite:
                invite.submission_id = new_sub.id
                invite.status = "form_received"
                db.session.commit()

            count += 1

        return render_template("fetch_invite_submissions.html", count=count)



    @app.route("/process_all_invites")
    def process_all_invites():
        # Only submissions that are not processed
        pending = (
            db.session.query(IncomingSubmission)
            .filter_by(processed=False)
            .all()
        )

        count = 0

        for sub in pending:
            # Try to match to an invite
            invite = match_submission_to_invite(sub)
            if not invite:
                continue

            # Only process invites that are awaiting_form
            if invite.status != "awaiting_form":
                continue

            # Mark invite as form_received
            invite.status = "form_received"
            db.session.commit()

            # Process the submission (creates lesson, attaches riders, etc.)
            process_submission(sub)
            count += 1

        return f"Processed {count} invite submissions."

    @app.route("/clear_processed_invites")
    def clear_processed_invites():
        processed = (
            db.session.query(IncomingSubmission)
            .filter_by(processed=True)
            .all()
        )

        count = len(processed)

        for sub in processed:
            db.session.delete(sub)

        db.session.commit()

        return f"Cleared {count} processed invite submissions."





    @app.route('/blockout_dates', methods=['GET','POST'])
    def manage_blockout_dates():
        if request.method == 'POST':
            raw_date = request.form.get('block_date')
            reason = request.form.get('reason','').strip()
            block_date = datetime.strptime(raw_date, '%Y-%m-%d').date()
            db.session.add(BlockoutDate(block_date=block_date, reason=reason))
            db.session.commit()
            flash(f"Blockout added: {block_date} ({reason})", "success")
        rows = db.session.query(BlockoutDate).order_by(BlockoutDate.block_date).all()
        return render_template('blockout_dates.html', blockouts=rows)

    @app.route('/blockout_dates/delete/<int:row_id>', methods=['POST'])
    def delete_blockout_date(row_id):
        row = db.session.query(BlockoutDate).get(row_id)
        if row:
            db.session.delete(row)
            db.session.commit()
            flash("Blockout removed.", "success")
        return redirect(url_for('manage_blockout_dates'))


    @app.route('/blockout_ranges', methods=['GET','POST'])
    def manage_blockout_ranges():
        if request.method == 'POST':
            start_raw = request.form.get('start_date')
            end_raw = request.form.get('end_date')
            reason = request.form.get('reason','').strip()
            start_date = datetime.strptime(start_raw, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_raw, '%Y-%m-%d').date()
            db.session.add(BlockoutRange(start_date=start_date, end_date=end_date, reason=reason))
            db.session.commit()
            flash(f"Blockout range added: {start_date} → {end_date} ({reason})", "success")
        rows = db.session.query(BlockoutRange).order_by(BlockoutRange.start_date).all()
        return render_template('blockout_ranges.html', blockouts=rows)

    @app.route('/blockout_ranges/delete/<int:row_id>', methods=['POST'])
    def delete_blockout_range(row_id):
        row = db.session.query(BlockoutRange).get(row_id)
        if row:
            db.session.delete(row)
            db.session.commit()
            flash("Blockout range removed.", "success")
        return redirect(url_for('manage_blockout_ranges'))


    @app.route("/new_lesson", methods=["POST"])
    def new_lesson():
        import re
        from datetime import datetime, timedelta, date

        lesson_date_str = request.form.get("date")
        lesson_id       = request.form.get("lesson_id")
        start_raw       = request.form.get("start", "")
        end_raw         = request.form.get("end", "")
        price_pl        = request.form.get("price_pl", "$0.00")
        lesson_type     = request.form.get("lesson_type", "Arena")
        group_priv      = request.form.get("group_priv", "")
        freq            = request.form.get("freq", "")
        respect_blockouts = request.form.get("respect_blockouts", "yes")

        client_mode   = request.form.get("client_mode", "existing")
        client_id_raw = request.form.get("client_id")
        client_name   = request.form.get("client_name", "").strip()
        client_phone  = request.form.get("client_phone", "").strip()
        horse         = request.form.get("horse", "").strip()

        print("DEBUG FORM:", dict(request.form))

        def _parse_times(start_label, end_form):
            if end_form and str(end_form).strip():
                start_token = (str(start_label).split('-')[0]).strip() if start_label else ''
                return start_token, str(end_form).strip()

            m = re.match(r'^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', str(start_label or ''))
            if m:
                return m.group(1), m.group(2)

            return (str(start_label or '').strip(), str(end_form or '').strip())

        def _clean_time_range(time_range):
            parts = [p.strip() for p in str(time_range or "").split('-') if p.strip()]
            if len(parts) >= 2:
                return f"{parts[0]} - {parts[-1]}"
            return str(time_range or "").strip()

        # ---------------------------------------------------------
        # EDIT EXISTING LESSON
        # ---------------------------------------------------------
        if lesson_id:
            lesson = Lesson.query.get(int(lesson_id))
            if not lesson:
                return redirect(url_for("lessons_by_date", date=lesson_date_str))

            end_effective   = end_raw.strip()
            start_effective = start_raw.strip()

            if not end_effective:
                existing_tf = lesson.time_frame or ""
                m = re.match(r'^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', existing_tf)
                if m:
                    if not start_effective:
                        start_effective = m.group(1)
                    end_effective = m.group(2)

            start_token, end_token = _parse_times(start_effective or lesson.time_frame, end_effective)
            time_range = _clean_time_range(f"{start_token} - {end_token}")
            time_frame = time_range

            existing_range = db.session.query(Time).filter_by(timerange=time_range).first()
            if not existing_range:
                db.session.add(Time(timerange=time_range))
                db.session.commit()

            if client_mode == "new":
                if client_name:
                    new_client = Client(full_name=client_name, mobile=client_phone)
                    db.session.add(new_client)
                    db.session.flush()
                    client_id = new_client.client_id
                else:
                    client_id = lesson.client_id
            else:
                client_id = int(client_id_raw) if client_id_raw and str(client_id_raw).isdigit() else lesson.client_id

            lesson.lesson_date = datetime.strptime(lesson_date_str, "%Y-%m-%d").date() if lesson_date_str else lesson.lesson_date
            lesson.time_frame  = time_frame
            lesson.price_pl    = parse_money(price_pl)
            lesson.lesson_type = lesson_type
            lesson.group_priv  = group_priv
            lesson.freq        = freq
            lesson.horse       = horse

            db.session.commit()

            # Recalculate this client's entire ledger after editing
            if client_id:
                recalc_client_lessons_by_id(client_id)

            return redirect(url_for("lessons_by_date", date=lesson_date_str))

        # ---------------------------------------------------------
        # CREATE NEW LESSON(S)
        # ---------------------------------------------------------
        start_token, end_token = _parse_times(start_raw, end_raw)
        time_range = _clean_time_range(f"{start_token} - {end_token}")

        if not start_token or not end_token or "-" not in time_range:
            print("[DEBUG] Missing or invalid time range for NEW lesson, rejecting.")
            return redirect(url_for("lessons_by_date", date=lesson_date_str))

        time_frame = time_range

        existing_range = db.session.query(Time).filter_by(timerange=time_range).first()
        if not existing_range:
            db.session.add(Time(timerange=time_range))
            db.session.commit()

        # Determine client_id
        if client_mode == "new":
            if client_name:
                new_client = Client(full_name=client_name, mobile=client_phone)
                db.session.add(new_client)
                db.session.flush()
                client_id = new_client.client_id
            else:
                client_id = None
        else:
            client_id = int(client_id_raw) if client_id_raw and str(client_id_raw).isdigit() else None

        # Canonical client name (CRITICAL FIX)
        if client_id:
            client_obj = Client.query.get(client_id)
            canonical_name = client_obj.full_name if client_obj else client_name
        else:
            canonical_name = client_name

        # Blockouts
        block_dates  = {b.block_date for b in db.session.query(BlockoutDate).all()}
        block_ranges = [(r.start_date, r.end_date) for r in db.session.query(BlockoutRange).all()]

        def is_blocked(d):
            if d in block_dates:
                return True
            for start_d, end_d in block_ranges:
                if start_d <= d <= end_d:
                    return True
            return False

        start_date = datetime.strptime(lesson_date_str, "%Y-%m-%d").date()
        end_of_year = date(start_date.year, 12, 31)

        if freq == "S":
            dates = [start_date]
        elif freq == "F":
            dates = []
            d = start_date
            while d <= end_of_year:
                dates.append(d)
                d += timedelta(days=14)
        elif freq == "W":
            dates = []
            d = start_date
            while d <= end_of_year:
                dates.append(d)
                d += timedelta(days=7)
        else:
            dates = [start_date]

        added = 0
        for d in dates:
            blocked = is_blocked(d)
            print(f"[DEBUG] candidate={d} blocked={blocked} respect_blockouts={respect_blockouts}")

            if respect_blockouts == "yes" and blocked:
                print(f"[DEBUG] skipped {d} due to blockout")
                continue

            lesson = Lesson(
                lesson_date=d,
                time_frame=time_frame,
                price_pl=parse_money(price_pl),
                lesson_type=lesson_type,
                group_priv=group_priv,
                freq=freq,
                client=canonical_name,
                horse=horse,
            )
            db.session.add(lesson)
            added += 1
            print(f"[DEBUG] added lesson for {d} client={canonical_name} horse={horse}")

        db.session.commit()
        print(f"[DEBUG] commit done, total lessons added={added}")

        # Recalculate this client's entire ledger
        if client_id:
            recalc_client_lessons_by_id(client_id)

        return redirect(url_for("lessons_by_date", date=lesson_date_str))



    @app.route('/delete_client_lessons/<int:client_id>', methods=['POST'])
    def delete_client_lessons(client_id):
        mode = (request.form.get('mode') or '').strip()
        group_priv = (request.form.get('group_priv') or '').strip()
        cutoff_str = (request.form.get('cutoff_date') or '').strip()

        # Default to today if no date provided
        cutoff_date = date.today()
        if cutoff_str:
            try:
                cutoff_date = datetime.strptime(cutoff_str, "%Y-%m-%d").date()
            except ValueError:
                flash("Invalid cutoff date format, using today instead.", "warning")

        client = db.session.query(Client).get(client_id)
        if not client:
            flash("Client not found.", "danger")
            return redirect(url_for('debug_page'))

        if mode == "all":
            db.session.query(Lesson).filter(
                func.lower(func.trim(Lesson.client)) == func.lower(func.trim(client.full_name)),
                Lesson.lesson_date >= cutoff_date
            ).delete(synchronize_session=False)

        elif mode == "group_priv" and group_priv:
            db.session.query(Lesson).filter(
                func.lower(func.trim(Lesson.client)) == func.lower(func.trim(client.full_name)),
                Lesson.lesson_date >= cutoff_date,
                func.lower(func.trim(Lesson.group_priv)) == group_priv.lower()
            ).delete(synchronize_session=False)

        else:
            flash("Invalid delete request.", "danger")
            return redirect(url_for('client_view', client=client.full_name))

        db.session.commit()
        flash(f"Delete executed for {client.full_name} from {cutoff_date}", "success")
        log_admin_action(f"Delete executed for {client.full_name} from {cutoff_date}", user="system")
        return redirect(url_for('client_view', client=client.full_name))




    @app.route('/admin/logs')
    def view_admin_logs():
        log_dir = LOG_ROOT
        logs = []
        try:
            # Collect all admin log files
            for fname in sorted(os.listdir(log_dir), reverse=True):
                if fname.startswith("admin_actions_") and fname.endswith(".log"):
                    path = os.path.join(log_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        logs.append((fname, f.read()))
        except Exception as e:
            logs.append(("Error", f"Failed to read logs: {e}"))
        return render_template("admin_logs.html", logs=logs)


    @app.route('/admin/reindex', methods=['POST'])
    def reindex_tables():
        try:
            tables = ['lessons', 'clients', 'teacher_time']
            for table in tables:
                db.session.execute(text(f'REINDEX TABLE {table};'))
            db.session.commit()
            flash('Reindexing completed successfully.', 'success')
            log_admin_action(f"Reindexed tables: {', '.join(tables)}", user="system")
        except Exception as e:
            db.session.rollback()
            flash(f'Reindexing failed: {str(e)}', 'danger')
            log_admin_action(f"Reindex failed: {str(e)}", user="system")
        return redirect(url_for('debug_page'))


    @app.route('/send_invoice')
    def send_invoice():
        import win32com.client
        from datetime import datetime

        lesson_id = request.args.get('lesson_id')
        if not lesson_id:
            return "Missing lesson ID", 400

        lesson = db.session.query(Lesson).get(int(lesson_id))
        if not lesson:
            return "Lesson not found", 404

        client = db.session.query(Client).filter(
                func.lower(func.trim(Client.full_name)) == func.lower(func.trim(lesson.client))
        ).first()
        if not client:
            return "Client not found", 404

        # Format email
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.To = client.email_primary or client.guardian_contact or "your@email.com"
        mail.Subject = f"Invoice for {lesson.lesson_date.strftime('%d %b %Y')}"
        mail.Body = f"""Hi {client.guardian_name},

Please find your invoice for the lesson on {lesson.lesson_date.strftime('%A, %d %B %Y')}.

Client: {client.full_name}
Horse: {lesson.horse}
Time: {lesson.time_frame}
Amount: ${lesson.price_pl:.2f}

Let us know if you have any questions.

Thanks,
Cherbon Waters Admin
"""
        mail.Display()  # Use mail.Send() to send immediately

        return "Invoice email opened in Outlook"




    @app.route('/update_invoice_flag', methods=['POST'])
    def update_invoice_flag():
        cid = int(request.form.get('client_id'))
        flag = 'invoice_required' in request.form
        client = db.session.query(Client).get(cid)
        if client:
            client.invoice_required = flag
            db.session.commit()
        return redirect(url_for('client_view', client=client.full_name))


    @app.route('/recalculate_all', methods=['POST'])
    def recalculate_all():
        clients = db.session.query(Lesson.client).distinct().all()
        for (client,) in clients:
            lessons = Lesson.query.filter_by(client=client).order_by(
                Lesson.lesson_date.asc(), Lesson.lesson_id.asc()
            ).all()
            bal = 0
            for l in lessons:
                att = (l.attendance or '').strip().upper()
                pay = l.payment or 0
                price = l.price_pl or 0
                if att in ['Y', 'N']:
                    l.balance = bal + pay - price
                else:
                    l.balance = bal + pay
                l.carry_fwd = bal
                bal = l.balance
        db.session.commit()
        flash("All lessons recalculated successfully.", "success")
        return redirect(url_for('client_view', client_filter=client), code=303)

    @app.route('/recalculate_client', methods=['POST'])
    def recalculate_client():
        client = request.form.get('client')
        lessons = Lesson.query.filter_by(client=client).order_by(
            Lesson.lesson_date.asc(), Lesson.lesson_id.asc()
        ).all()
        bal = 0
        for l in lessons:
            att = (l.attendance or '').strip().upper()
            pay = l.payment or 0
            price = l.price_pl or 0
            if att in ['Y', 'N']:
                l.balance = bal + pay - price
            else:
                l.balance = bal + pay
            l.carry_fwd = bal
            bal = l.balance
        db.session.commit()
        if lessons:
            flash(f"Lessons recalculated for client: {client}", "success")
        else:
            flash(f"No lessons found for client: {client}", "danger")
        return redirect(url_for('client_view', client_filter=client), code=303)

    @app.route('/notifications/clear_processed')
    def clear_processed():
        db.session.query(IncomingSubmission).filter(
            (IncomingSubmission.processed == True) |
            (IncomingSubmission.processed.is_(None))
        ).delete(synchronize_session=False)
        db.session.commit()
        return redirect(url_for('notifications'))


    @app.route('/manage_teacher_times', methods=['GET'])
    def manage_teacher_times():
        rows = db.session.query(TeacherTime).order_by(
            TeacherTime.teacher_key,
            TeacherTime.weekday,
            TeacherTime.time
        ).all()

        grouped = {}
        for r in rows:
            grouped.setdefault(r.teacher_key, {}).setdefault(r.weekday, []).append(r)

        teacher_keys = [f"Teacher {i}" for i in range(1, 6)]
        for teacher in teacher_keys:
            grouped.setdefault(teacher, {})
            for weekday in range(7):
                grouped[teacher].setdefault(weekday, [])

        return render_template('manage_teacher_times_grouped.html',
                               grouped=grouped,
                               teacher_keys=teacher_keys)

    @app.route('/manage_teacher_times/add', methods=['POST'])
    def add_teacher_time():
        teacher_key = (request.form.get('teacher_key') or '').strip()
        weekday = request.form.get('weekday')
        time_input = (request.form.get('time') or '').strip()

        if not teacher_key or not weekday or not time_input:
            flash("Missing teacher, weekday, or time.", "danger")
            return redirect(url_for('manage_teacher_times'))

        # Normalize separators (e.g., "11.00" -> "11:00", "11 00" -> "11:00")
        raw_time = time_input
        time_norm = re.sub(r'[.\s]+', ':', raw_time)

        # Validate basic HH:MM shape
        m = re.match(r'^(\d{1,2}):(\d{1,2})$', time_norm)
        if not m:
            flash(f"Invalid time format: '{raw_time}'. Use HH:MM (e.g., 09:30).", "danger")
            return redirect(url_for('manage_teacher_times'))

        hour = int(m.group(1))
        minute = int(m.group(2))

        # Bounds check
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            flash(f"Out-of-range time: '{raw_time}'. Use 00:00 to 23:59.", "danger")
            return redirect(url_for('manage_teacher_times'))

        # Zero-pad to canonical HH:MM
        time_val = f"{hour:02d}:{minute:02d}"

        try:
            weekday_int = int(weekday)
        except Exception:
            weekday_int = 0

        t = TeacherTime(teacher_key=teacher_key, weekday=weekday_int, time=time_val)
        db.session.add(t)
        db.session.commit()
        flash(f"Added {teacher_key} - weekday {weekday_int} at {time_val}.", "success")
        return redirect(url_for('manage_teacher_times'))

    @app.route('/manage_teacher_times/delete/<int:row_id>', methods=['POST'])
    def delete_teacher_time(row_id):
        row = db.session.query(TeacherTime).get(row_id)
        if row:
            db.session.delete(row)
            db.session.commit()
        return redirect(url_for('manage_teacher_times'))







    @app.route("/debug_dates")
    def debug_dates():
        out = []
        for l in Lesson.query.order_by(Lesson.lesson_id.asc()).all():
            out.append(f"{l.lesson_id} | {repr(l.client)} | {repr(l.lesson_date)} | {type(l.lesson_date)}")
        return "<br>".join(out)


    @app.route('/update_lessons', methods=['POST'], endpoint='update_lessons')
    def update_lessons():
        editable_fields = {
            'horse': lambda v: (v.strip() or None),
            'attendance': lambda v: (v.strip() or None),
            'payment': lambda v: (
                parse_money(v) if v and str(v).strip() not in ['0', '0.0', '0.00', '$0.00']
                else None
            ),
            'price_pl': lambda v: (parse_money(v) if v and str(v).strip() else None),
            'bank_transfer': lambda v: (v or '').strip(),
        }

        # collect lesson IDs from form
        lesson_ids = {
            int(m.group(1))
            for key in request.form.keys()
            if (m := re.match(r'.*_(\d+)$', key))
        }

        selected_date, selected_date_str = parse_selected_date()
        if not lesson_ids and not selected_date:
            return redirect(url_for('lessons_by_date'))

        lessons_q = db.session.query(Lesson).filter(Lesson.lesson_id.in_(lesson_ids)).all()
        lessons_by_id = {l.lesson_id: l for l in lessons_q}

        changes = {}  # collect changes for logging

        for lid in sorted(lesson_ids):
            lesson = lessons_by_id.get(lid)
            if not lesson:
                continue
            for field, caster in editable_fields.items():
                form_key = f"{field}_{lid}"
                if form_key in request.form:
                    raw = request.form.get(form_key, '')
                    try:
                        new_val = caster(raw)
                    except Exception:
                        continue
                    old_val = getattr(lesson, field, None)
                    if old_val != new_val:
                        setattr(lesson, field, new_val)
                        changes.setdefault(lid, {})[field] = (old_val, new_val)

            # Recalculate this lesson's balance
            att = (lesson.attendance or '').strip().upper()
            pay = lesson.payment or 0
            price = lesson.price_pl or 0
            carry = lesson.carry_fwd or 0

            if att in ['', 'C']:
                lesson.balance = carry + pay
            elif att in ['Y', 'N']:
                lesson.balance = carry + pay - price

            # Recalculate all future lessons for this client
            future_lessons = db.session.query(Lesson).filter(
                Lesson.client == lesson.client,
                Lesson.lesson_date > lesson.lesson_date
            ).order_by(Lesson.lesson_date.asc(), Lesson.lesson_id.asc()).all()

            bal = lesson.balance
            for l in future_lessons:
                att = (l.attendance or '').strip().upper()
                pay = l.payment or 0
                price = l.price_pl or 0

                if att in ['Y', 'N']:
                    l.balance = bal + pay - price
                else:
                    l.balance = bal + pay

                l.carry_fwd = bal
                bal = l.balance

        # Update client notes
        client_ids = {
            int(key.rsplit('_', 1)[1])
            for key in request.form.keys()
            if key.startswith("notes_") and key.rsplit('_', 1)[1].isdigit()
        }

        clients = db.session.query(Client).filter(Client.client_id.in_(client_ids)).all()
        clients_by_id = {c.client_id: c for c in clients}

        for cid in client_ids:
            client = clients_by_id.get(cid)
            if not client:
                continue
            form_key = f"notes_{cid}"
            new_notes = request.form.get(form_key, '').strip()
            old_notes = (client.notes or '').strip()
            new_notes = request.form.get(form_key, '').strip()

            if old_notes != new_notes:
                changes.setdefault(f"client_{cid}", {})["notes"] = (client.notes, new_notes)
                client.notes = new_notes

        # teacher block tags
        grouped_lookup = {}
        if selected_date:
            lesson_rows = db.session.query(Lesson).filter_by(
                lesson_date=selected_date
            ).order_by(Lesson.time_frame).all()
            time_lookup = {norm(t.timerange): t for t in db.session.query(Time).all()}
            for lesson in lesson_rows:
                time_key = norm(lesson.time_frame)
                time_obj = time_lookup.get(time_key)
                if time_obj and lesson.lesson_type and lesson.group_priv:
                    key = norm_timerange_key(time_obj.timerange)
                    if key not in grouped_lookup:
                        grouped_lookup[key] = (lesson.lesson_type, lesson.group_priv)

        # Build tag lists from form; sentinel hidden input ensures key is present even when no checkboxes checked
        block_tags = {}
        for key in request.form:
            if key.startswith('teacher_tag_'):
                block_key = key[len('teacher_tag_'):]
                raw_list = request.form.getlist(key)
                # keep the key even if only empties
                tag_list = [t for t in raw_list if t.strip()]
                block_tags[block_key] = tag_list






        # DEBUG: inspect what we received from the form (temporary)
        print("DEBUG block_tags from form:", block_tags)

        # Persist tags: empty list becomes empty string in DB (explicitly clearing saved tags)
        if selected_date:
            for block_key, tag_list in block_tags.items():
                lesson_type, group_priv = grouped_lookup.get(block_key, ('Unknown', 'Unknown'))

                # ✅ normalize block_key to fit varchar(20)
                safe_time_range = norm_timerange_key(block_key)[:20]

                existing = db.session.query(LessonBlockTag).filter_by(
                    lesson_date=selected_date,
                    time_range=safe_time_range
                ).first()

                if existing:
                    existing.teacher_tags = ','.join(tag_list)  # '' if tag_list == []
                    existing.lesson_type = lesson_type
                    existing.group_priv = group_priv
                else:
                    new_tag = LessonBlockTag(
                        lesson_date=selected_date,
                        time_range=safe_time_range,   # 👈 normalized key
                        lesson_type=lesson_type,
                        group_priv=group_priv,
                        teacher_tags=','.join(tag_list)
                    )
                    db.session.add(new_tag)
        # commit DB changes (lessons, clients, and block tags)
        try:
            db.session.commit()
            print("DEBUG commit done for date:", selected_date)
        except Exception as e:
            db.session.rollback()
            print("DB commit failed:", e)
            flash("Save failed: " + str(e), "danger")
            return redirect(url_for('lessons_by_date', date=selected_date_str), code=303)

        try:
            db.session.commit()
        except Exception as e:
            print(f"Commit failed: {e}")
            db.session.rollback()
            return redirect(url_for('lessons_by_date'), code=303)

        # Log changes separately so logging errors don't affect committed data
        if changes:
            try:
                log_lesson_changes(changes, user='system')  # no current_user dependency
                print("Logged changes:", changes)
            except Exception as e:
                print(f"Log write failed: {e}")

        return redirect(url_for('lessons_by_date', date=selected_date_str), code=303)


    @app.route('/admin/seed_jotform_ids')
    def seed_jotform_ids():
        rows = db.session.query(IncomingSubmission).all()
        updated = 0

        for row in rows:
            submission_id = str(row.form_id)
            riders = parse_jotform_payload(row.raw_payload)

            for rider in riders:
                # Use your existing fuzzy matching logic
                matches = rider.get("matches", [])
                existing = matches[0] if matches else None

                if existing:
                    # Seed the JotForm submission ID
                    existing.jotform_submission_id = submission_id
                    updated += 1

        db.session.commit()
        return f"Seed complete. Updated {updated} clients."


    @app.route('/save_txt', methods=['POST'])
    def save_txt():
        try:
            selected_date = request.args.get("selected_date")
            if not selected_date:
                return {"error": "Missing selected_date"}, 400

            day = datetime.strptime(selected_date, "%Y-%m-%d").date()
            today_str = day.strftime("%d-%m-%y")
            filename = f"lesson_schedule_{today_str}"

            # --- Pull all horses (ALWAYS include all horses) ---
            horses = [
                (h.horse or "").strip()
                for h in db.session.query(Horse).order_by(Horse.horse).all()
                if (h.horse or "").strip()
            ]

            # --- Pull lessons for the day ---
            lessons = db.session.query(Lesson).filter(
                Lesson.lesson_date == day
            ).order_by(Lesson.time_frame.asc()).all()

            # --- Build time slots from lessons ---
            time_slots = sorted({
                (l.time_frame or "").split("-")[0].strip()
                for l in lessons
                if l.time_frame
            })

            # --- Build empty schedule matrix ---
            schedule = {h: {slot: "" for slot in time_slots} for h in horses}

            # --- Fill schedule (skip attendance C) ---
            for l in lessons:
                h = (l.horse or "").strip()
                if not h:
                    continue
                if h not in schedule:
                    continue

                if (l.attendance or "").upper() == "C":
                    continue  # cancelled → blank cell

                slot = (l.time_frame or "").split("-")[0].strip()
                if slot not in time_slots:
                    continue

                # Display rules
                if (l.lesson_type or "").strip() == "Trail Ride":
                    disp = f"{slot}T"
                else:
                    disp = slot

                schedule[h][slot] = disp

            # --- Build TXT output ---
            columns = ["Horse"] + time_slots

            # Column widths
            col_widths = []
            for col in columns:
                if col == "Horse":
                    max_len = max(len(h) for h in horses)
                else:
                    max_len = max(len(schedule[h].get(col, "")) for h in horses)
                    max_len = max(max_len, len(col))
                col_widths.append(max(max_len, 5))

            def fmt_row(cells):
                return "| " + " | ".join(
                    str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
                ) + " |"

            header = f"Lesson Schedule for {today_str}"
            header_line = fmt_row(columns)
            underline = "|" + "|".join("_" * (w + 2) for w in col_widths) + "|"

            lines = [header, "", header_line, underline]

            for h in horses:
                row = [h] + [schedule[h][slot] for slot in time_slots]
                lines.append(fmt_row(row))
                lines.append(underline)

            full_text = "\n".join(lines)

            # --- Write TXT ---
            out_dir = CSV_ROOT            
            txt_path = os.path.join(out_dir, f"{filename}.txt")
            excel_path = os.path.join(out_dir, f"{filename}.xlsx")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # --- Excel Output ---
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, Border, Side

            wb = Workbook()
            ws = wb.active
            ws.title = "Lesson Schedule"

            # Header row
            ws.append(columns)

            # Data rows
            for h in horses:
                row = [h] + [schedule[h][slot] for slot in time_slots]
                ws.append(row)

            # Styling
            bold_font = Font(bold=True)
            center_align = Alignment(horizontal='center', vertical='center')
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            for row in ws.iter_rows():
                for cell in row:
                    cell.font = bold_font
                    cell.alignment = center_align
                    cell.border = thin_border

            # Page setup
            ws.page_margins.left   = 0.2
            ws.page_margins.right  = 0.2
            ws.page_margins.top    = 0.1
            ws.page_margins.bottom = 0.1
            ws.page_margins.header = 0.0
            ws.page_margins.footer = 0.0
            ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
            ws.page_setup.paperSize = ws.PAPERSIZE_A4

            # Auto-fit columns
            for col in ws.columns:
                max_length = max(len(str(cell.value or "")) for cell in col)
                adjusted_width = max(max_length + 2, 10)
                ws.column_dimensions[col[0].column_letter].width = adjusted_width

            wb.save(excel_path)

            # --- Launch Notepad++ ---
            
            return {"status": "Text and Excel files saved successfully"}, 200

        except Exception as e:
            return {"error": str(e)}, 500

    @app.route("/debug_order/<client>")
    def debug_order(client):
        lessons = Lesson.query.filter_by(client=client).order_by(
            Lesson.lesson_date.asc(),
            Lesson.lesson_id.asc()
        ).all()

        out = []
        for l in lessons:
            out.append(f"{l.lesson_id} | {l.lesson_date} | {l.time_frame}")
        return "<br>".join(out)



    @app.route('/debug_payload/<int:id>')
    def debug_payload(id):
        row = db.session.query(IncomingSubmission).get(id)
        print(row.raw_payload)
        return "Printed to console"

    @app.route("/client/<int:client_id>/change_price", methods=["POST"])
    def change_client_price(client_id):
        mode = request.form.get("mode")
        group_priv = request.form.get("group_priv")
        cutoff_date = request.form.get("cutoff_date")
        new_price = request.form.get("new_price")

        if not group_priv:
            flash("Group Priv is required.", "danger")
            return redirect(url_for("client_view", client_id=client_id))

        if not new_price:
            flash("New price is required.", "danger")
            return redirect(url_for("client_view", client_id=client_id))

        # Update price_pl for all matching lessons from cutoff date
        db.session.execute("""
            UPDATE lessons
            SET price_pl = :new_price
            WHERE client_id = :client_id
              AND group_priv = :group_priv
              AND lesson_date >= :cutoff_date
        """, {
            "new_price": new_price,
            "client_id": client_id,
            "group_priv": group_priv,
            "cutoff_date": cutoff_date
        })

        db.session.commit()
        return redirect(url_for("client_view", client_id=client_id))


    @app.route("/client/<int:client_id>/change_horse", methods=["POST"])
    def change_client_horse(client_id):
        mode = request.form.get("mode")
        group_priv = request.form.get("group_priv")
        cutoff_date = request.form.get("cutoff_date")
        new_horse = request.form.get("new_horse")

        if not group_priv:
            flash("Group Priv is required.", "danger")
            return redirect(url_for("client_view", client_id=client_id))

        if mode == "change_horse":
            # Change horse for ALL matching lessons from cutoff date
            db.session.execute("""
                UPDATE lessons
                SET horse_id = :new_horse
                WHERE client_id = :client_id
                  AND group_priv = :group_priv
                  AND lesson_date >= :cutoff_date
            """, {
                "new_horse": new_horse,
                "client_id": client_id,
                "group_priv": group_priv,
                "cutoff_date": cutoff_date
            })

        elif mode == "assign_if_empty":
            # Assign horse ONLY where horse_id is NULL or empty
            db.session.execute("""
                UPDATE lessons
                SET horse_id = :new_horse
                WHERE client_id = :client_id
                  AND group_priv = :group_priv
                  AND lesson_date >= :cutoff_date
                  AND (horse_id IS NULL OR horse_id = '')
            """, {
                "new_horse": new_horse,
                "client_id": client_id,
                "group_priv": group_priv,
                "cutoff_date": cutoff_date
            })

        db.session.commit()
        return redirect(url_for("client_view", client_id=client_id))


    @app.route('/client_view')
    def client_view():
        client_id = request.args.get('client', type=int) or request.args.get('client_filter', type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = int(request.args.get('page', 1))
        per_page = 50

        lessons = []
        total = 0
        client_obj = None

        # ---------------------------------------------------------
        # LOAD ALL CLIENTS
        # ---------------------------------------------------------
        clients = db.session.query(Client).order_by(Client.full_name.asc()).all()
        client_names = [(c.client_id, c.full_name) for c in clients]

        # ---------------------------------------------------------
        # LOAD ALL HORSES  <-- THIS WAS MISSING
        # ---------------------------------------------------------
        horses = db.session.query(Horse).order_by(Horse.horse.asc()).all()

        # ---------------------------------------------------------
        # LOOK UP SELECTED CLIENT
        # ---------------------------------------------------------
        if client_id:
            client_obj = Client.query.get(client_id)

            if client_obj:
                # Filter lessons by exact client name match
                query = db.session.query(Lesson).filter(
                    Lesson.client == client_obj.full_name
                )

                if start_date:
                    query = query.filter(Lesson.lesson_date >= start_date)
                if end_date:
                    query = query.filter(Lesson.lesson_date <= end_date)

                total = query.count()

                # ---------------------------------------------------------
                # ORDER BY TRUE DATE ASCENDING (SQL-LEVEL DATE CAST)
                # ---------------------------------------------------------
                lessons = (
                    query.order_by(
                        db.func.date(Lesson.lesson_date).desc(),
                        Lesson.lesson_id.desc()
                    )
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )

        return render_template(
            'client_view.html',
            lessons=lessons,
            client_filter=client_id,
            start_date=start_date,
            end_date=end_date,
            page=page,
            total=total,
            per_page=per_page,
            client_names=client_names,
            client_obj=client_obj,
            horses=horses,                     # <-- NOW INCLUDED
            today=date.today().isoformat()
        )

    return app


app = create_app()


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)