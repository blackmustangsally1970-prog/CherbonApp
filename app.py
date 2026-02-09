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
from sqlalchemy.orm import joinedload




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



def recalc_all_lessons():
    """
    Recalculate per-lesson balances for ALL lessons.
    Simple model:
        balance = carry_fwd + payment + adjust - price_pl (if attended/charged)
    """
    lessons = (
        Lesson.query
        .order_by(
            db.func.date(Lesson.lesson_date).asc(),
            Lesson.lesson_id.asc()
        )
        .all()
    )

    for l in lessons:
        carry = l.carry_fwd or 0
        payment = l.payment or 0
        price = l.price_pl or 0
        adjust = l.adjust or 0
        att = (l.attendance or '').strip().upper()

        balance = carry + payment + adjust
        if att in ['Y', 'N']:
            balance -= price

        l.balance = balance

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

# ---------------------------------------------------------
# HELPERS (NO INDENT)
# ---------------------------------------------------------

def handle_existing_client(data):
    client_id = data.get("existing_client_id")
    lessons = data.get("lessons", [])

    if not client_id:
        return jsonify(success=False, error="No client selected")

    try:
        for l in lessons:
            lesson = Lesson(
                lesson_date = l.get("date"),
                time_frame  = l.get("time"),
                lesson_type = l.get("type"),
                group_priv  = l.get("grouppriv"),
                price_pl    = l.get("price"),
                client_id   = client_id,
                client      = "",
                horse       = "",
                attendance  = "Pending",
                payment     = None
            )
            db.session.add(lesson)

        db.session.commit()
        return jsonify(success=True)

    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, error=str(e))


def handle_new_client(data):
    name   = data.get("new_client_name")
    mobile = data.get("mobile")
    lessons = data.get("lessons", [])

    if not name:
        return jsonify(success=False, error="Client name required")

    try:
        client = Client(
            full_name = name,
            mobile    = mobile,
            notes     = "Created via booking panel"
        )
        db.session.add(client)
        db.session.flush()

        for l in lessons:
            lesson = Lesson(
                lesson_date = l.get("date"),
                time_frame  = l.get("time"),
                lesson_type = l.get("type"),
                group_priv  = l.get("grouppriv"),
                price_pl    = l.get("price"),
                client_id   = client.client_id,
                client      = name,
                horse       = "",
                attendance  = "Pending",
                payment     = None
            )
            db.session.add(lesson)

        db.session.commit()
        return jsonify(success=True)

    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, error=str(e))


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



def parse_jotform_payload(payload, forced_submission_id=None, clients_cache=None, mode="full"):
    """
    mode:
      - "light": parse names/fields only. NO DB client load. NO matching.
      - "full": full matching behaviour (may use clients_cache or DB load).
    clients_cache:
      - Optional preloaded list of Client rows (or tuples) to avoid repeated DB scans.
    """
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
            print("ERROR: Could not decode payload:", type(payload))
            return []

    # NOTE: you currently print full payloads here; consider gating by env var (can be huge / sensitive)
    # print("RAW PAYLOAD:", json.dumps(payload, indent=2))

    # Submission ID
    if forced_submission_id:
        submission_id = str(forced_submission_id)
    else:
        submission_id = str(
            payload.get("id")
            or payload.get("submission_id")
            or payload.get("form_id")
            or ""
        )

    answers = payload.get("answers", {}) or {}

    # ---- Email autodetect (keep your existing logic) ----
    email = ""
    for key, item in answers.items():
        if item.get("type") == "control_email":
            ans = item.get("answer")
            if isinstance(ans, dict):
                email = ans.get("value") or ans.get("text") or ans.get("full") or ""
            else:
                email = ans or ""
            break

    if not email:
        for key, item in answers.items():
            if item.get("name", "").lower() == "email":
                ans = item.get("answer")
                if isinstance(ans, dict):
                    email = ans.get("value") or ans.get("text") or ans.get("full") or ""
                else:
                    email = ans or ""
                break

    email = email or ""

    # Detect which form this submission belongs to
    form_id = str(payload.get("form_id") or "")
    is_invite_form = form_id == "253599154628066"

    # Invite token extraction (keep your existing logic)
    invite_token = None
    field3 = answers.get("3")
    if field3:
        ans = field3.get("answer")
        if isinstance(ans, dict):
            invite_token = ans.get("text") or ans.get("value") or ans.get("full")
        else:
            invite_token = ans

    if not invite_token:
        for f in answers.values():
            if f.get("name") == "i_t":
                ans = f.get("answer")
                if isinstance(ans, dict):
                    invite_token = ans.get("text") or ans.get("value") or ans.get("full")
                else:
                    invite_token = ans
                break

    if not invite_token:
        direct = answers.get("i_t")
        if direct:
            ans = direct.get("answer")
            if isinstance(ans, dict):
                invite_token = ans.get("text") or ans.get("value") or ans.get("full")
            else:
                invite_token = ans

    invite_token = invite_token or ""

    # Age field detection (your existing dynamic detection)
    age_fields = [
        key for key, item in answers.items()
        if item.get("type") in ("control_number", "control_dropdown")
        and "age" in item.get("text", "").lower()
    ]
    age_fields = sorted(age_fields, key=lambda x: int(x))

    # Global fields (kept consistent with your existing IDs)
    guardian = answers.get("86", {}).get("answer", "") or ""
    mobile = answers.get("87", {}).get("answer", {}).get("full", "") or ""
    email_fallback = answers.get("47", {}).get("answer", "") or ""
    disclaimer = answers.get("63", {}).get("answer", None)
    if email_fallback and not email:
        email = email_fallback

    def normalize_name(s):
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.replace("\xa0", " ")
        return " ".join(s.strip().lower().split())

    # Detect rider fullname fields (invite vs disclaimer)
    if is_invite_form:
        fullname_fields = INVITE_FULLNAME_FIELDS
    else:
        fullname_fields = [
            key for key, item in answers.items()
            if item.get("type") == "control_fullname"
            and item.get("text", "").lower().startswith("rider")
        ]
        fullname_fields = sorted(fullname_fields, key=lambda x: int(x))

    riders = []

    # LIGHT mode = no matching, no client cache needed
    do_matching = (mode == "full")

    client_cache = None
    exact_lookup = None

    if do_matching:
        # If not provided, fall back to DB load ONCE per call (still better than per-row per-page)
        if clients_cache is None:
            clients_cache = db.session.query(Client).all()

        client_cache = []
        for c in clients_cache:
            full_name = getattr(c, "full_name", None)
            if not full_name:
                continue
            client_cache.append((c, normalize_name(full_name), getattr(c, "jotform_submission_id", None)))

        exact_lookup = {norm: c for c, norm, _ in client_cache}

    for idx, fullname_key in enumerate(fullname_fields):
        item = answers.get(fullname_key)
        if not item:
            continue

        pretty = item.get("prettyFormat")
        if pretty:
            raw_name = pretty
        else:
            first = (item.get("answer", {}) or {}).get("first", "")
            last = (item.get("answer", {}) or {}).get("last", "")
            raw_name = f"{first} {last}"

        name_norm = normalize_name(raw_name)
        if not name_norm:
            continue

        age_key = age_fields[idx] if idx < len(age_fields) else None
        age = answers.get(age_key, {}).get("answer") if age_key else None

        rider = {
            "name": name_norm,
            "age": age,
            "guardian": guardian,
            "mobile": mobile,
            "email": email,
            "disclaimer": disclaimer,
            "matches": [],
            "jotform_submission_id": submission_id,
            "invite_token": invite_token
        }

        # Height/weight/notes - reuse your existing extract_number + field maps
        if is_invite_form:
            if idx < len(INVITE_HEIGHT_FIELDS):
                height_field = INVITE_HEIGHT_FIELDS[idx]
                weight_field = INVITE_WEIGHT_FIELDS[idx]
                notes_field = INVITE_NOTES_FIELDS[idx] if idx < len(INVITE_NOTES_FIELDS) else None
            else:
                height_field = weight_field = notes_field = None
        else:
            if idx < len(HEIGHT_FIELDS):
                height_field = HEIGHT_FIELDS[idx]
                weight_field = WEIGHT_FIELDS[idx]
                notes_field = NOTES_FIELDS[idx]
            else:
                height_field = weight_field = notes_field = None

        height_val = answers.get(height_field, {}).get("answer") if height_field else None
        weight_val = answers.get(weight_field, {}).get("answer") if weight_field else None
        notes_val = answers.get(notes_field, {}).get("answer") if notes_field else None

        rider["height_cm"] = extract_number(height_val)
        rider["weight_kg"] = extract_number(weight_val)
        rider["notes"] = notes_val or ""

        # Matching only in FULL mode (exact match only; keep listing cheap)
        if do_matching and exact_lookup is not None:
            matched_client = exact_lookup.get(name_norm)
            if matched_client:
                guardian_norm = normalize_name(guardian)
                if guardian_norm and normalize_name(matched_client.full_name) == guardian_norm:
                    pass
                elif matched_client.jotform_submission_id != submission_id:
                    rider["matches"].append(matched_client)

        riders.append(rider)

    return riders



def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "")
    app.config.from_object(Config())      # <-- fix
    # Load ClickSend credentials from Azure App Settings or environment
    app.config['CLICKSEND_USERNAME'] = (
        os.environ.get('CLICKSEND_USERNAME')
        or os.environ.get('APPSETTING_CLICKSEND_USERNAME')
        or ""
    )

    app.config['CLICKSEND_API_KEY'] = (
        os.environ.get('CLICKSEND_API_KEY')
        or os.environ.get('APPSETTING_CLICKSEND_API_KEY')
        or ""
    )


    app.config['SQLALCHEMY_ECHO'] = True
    db.init_app(app)

    # print("IncomingSubmission columns:", IncomingSubmission.__table__.columns.keys())  # <-- remove or wrap

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
            response = api_instance.sms_send_post(sms_messages)
            print("ClickSend API response:", response)

            try:
                http_code = int(getattr(response, "http_code", 0))
            except:
                http_code = 0

            if http_code == 200:
                return True

            print("ClickSend returned non-200:", response)
            return False

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
        clean_time = invite.time_frame
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


    @app.post("/recalculate_all")
    def recalculate_all():
        try:
            recalc_all_lessons()
            return jsonify(success=True, message="All lesson balances recalculated.")
        except Exception as e:
            return jsonify(success=False, error=str(e))


    @app.post("/update_lesson_field")
    def update_lesson_field():
        data = request.get_json()
        lesson_id = data.get("lesson_id")
        field = data.get("field")
        value = data.get("value")

        allowed = {"attendance", "carry_fwd", "payment", "price_pl", "adjust"}
        if field not in allowed:
            return jsonify(success=False, error="Invalid field")

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

        # no recalculation here
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

        # Build lookup: invite_token → submission model
        submission_lookup = {}
        for sub in submissions:

            # Ignore submissions from the disclaimer form
            # Your SMS invite form ID = 253599154628066
            if sub.form_id != "253599154628066":
                continue

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
            # If we found a matching submission, update invite status
            if sub and inv.status in ["waiting", "form_received"]:
                inv.status = "process"
                db.session.commit()

            parsed = None
            if sub:
                try:
                    riders = parse_jotform_payload(sub.raw_payload, forced_submission_id=sub.id)
                    parsed = riders[0] if riders else None
                except Exception:
                    parsed = None

            pending.append({
                "invite": inv,
                "submission": parsed,   # template‑safe dict
                "raw": sub              # optional model reference
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
        weekday_int = date.today().weekday()
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
        times = [t.timerange for t in db.session.query(Time).order_by(Time.timerange).all()]

        teacher_names = [t.teacher for t in get_static_teachers()]
        block_tag_lookup = {}
        invoice_clients = []  # ✅ ensure it’s always defined

        if selected_date:
            horse_schedule = defaultdict(list)
            lesson_rows = (
                db.session.query(Lesson)
                .filter_by(lesson_date=selected_date)
                .order_by(Lesson.time_frame)
                .all()
            )

            for lesson in lesson_rows:
                horse_name = to_proper_case(lesson.horse)
                att = (lesson.attendance or '').strip().upper()
                time_frame = (lesson.time_frame or '').strip()

                if horse_name and time_frame and att != 'C':
                    horse_schedule[horse_name].append(time_frame)

            time_lookup = {norm(t.timerange): t for t in db.session.query(Time).all()}

            client_lookup = {}
            clients = db.session.query(Client).all()

            for c in clients:
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

            for h, usage_times in teacher_horse_usage.items():
                teacher_horse_usage[h] = sorted(set(usage_times))



        print("ABOUT TO RENDER TEMPLATE — ALL DATA LOADED OK")


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
            times=times,   # 👈 add this
            get_static_teachers=get_static_teachers   # ⭐ ADD THIS
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
        clean_time = invite.time_frame

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

    def soft_match_client_for_invite(rider_name: str):
        """
        Soft‑match a rider name to existing clients.
        Returns:
            ("exact", client)
            ("none", None)
            ("ambiguous", [clients])
        """
        if not rider_name:
            return ("none", None)

        name_norm = (rider_name or "").strip().lower()

        # Exact match
        exact = db.session.query(Client).filter(
            func.lower(func.trim(Client.full_name)) == name_norm
        ).all()

        if len(exact) == 1:
            return ("exact", exact[0])
        if len(exact) > 1:
            return ("ambiguous", exact)

        # Soft match: remove spaces + hyphens
        compact = name_norm.replace(" ", "").replace("-", "")

        candidates = db.session.query(Client).filter(Client.full_name.isnot(None)).all()

        matches = []
        for c in candidates:
            c_name = (c.full_name or "").strip().lower()
            c_compact = c_name.replace(" ", "").replace("-", "")
            if c_compact == compact:
                matches.append(c)

        if not matches:
            return ("none", None)
        if len(matches) == 1:
            return ("exact", matches[0])
        return ("ambiguous", matches)


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

        clean_time = invite.time_frame

        for rider in riders:
            try:
                # ---------------------------------------------------------
                # STEP 3: Replace placeholder client + update pending lesson
                # ---------------------------------------------------------
                token = rider.get("invite_token")
                invite_for_rider = LessonInvite.query.filter_by(token=token).first()

                if invite_for_rider:
                    # 1. Fetch the pending lesson created at send_invite
                    pending_lesson = Lesson.query.get(invite_for_rider.lesson_id)

                    # 2. Fetch the placeholder client
                    placeholder_client = Client.query.filter_by(full_name=pending_lesson.client).first()

                    # 3. Resolve the REAL client using soft-match logic
                    rider_name = rider.get("full_name") or rider.get("name") or ""
                    match_type, match_data = soft_match_client_for_invite(rider_name)

                    if match_type == "exact":
                        real_client = match_data

                    elif match_type == "none":
                        # Create new client with ONLY name + height/weight
                        real_client = Client(
                            full_name=rider_name,
                            height_cm=rider.get("height_cm"),
                            weight_kg=rider.get("weight_kg"),
                            guardian_name="",
                            mobile="",
                            notes="Created via SMS invite"
                        )
                        db.session.add(real_client)
                        db.session.flush()

                    else:  # ambiguous
                        sub.needs_client_match = True
                        db.session.commit()
                        return "needs_client_match"

                    # 4. Update the pending lesson with REAL client details
                    pending_lesson.client = real_client.full_name
                    pending_lesson.height_cm = rider.get("height_cm")
                    pending_lesson.weight_kg = rider.get("weight_kg")
                    pending_lesson.horse = ""  # horse not assigned yet
                    pending_lesson.attendance = "Pending"
                    pending_lesson.payment = None

                    # 5. Mark invite as completed for this rider
                    invite_for_rider.status = "completed"

                    # 6. Delete placeholder client
                    db.session.delete(placeholder_client)

                    # 7. Add updated lesson to created list
                    created_lessons.append(pending_lesson)

                    # 8. Continue to next rider (no new lesson created here)
                    continue

                # ---------------------------------------------------------
                # STEP 4: Multi‑rider lesson cloning (rider #2+)
                # ---------------------------------------------------------
                if created_lessons:
                    # This is NOT the first rider — create a cloned lesson row

                    # Resolve client again (soft-match already done above)
                    rider_name = rider.get("full_name") or rider.get("name") or ""
                    match_type, match_data = soft_match_client_for_invite(rider_name)

                    if match_type == "exact":
                        real_client = match_data
                    elif match_type == "none":
                        real_client = Client(
                            full_name=rider_name,
                            height_cm=rider.get("height_cm"),
                            weight_kg=rider.get("weight_kg"),
                            guardian_name="",
                            mobile="",
                            notes="Created via SMS invite"
                        )
                        db.session.add(real_client)
                        db.session.flush()
                    else:
                        sub.needs_client_match = True
                        db.session.commit()
                        return "needs_client_match"

                    # Clone the first lesson's structure
                    base_lesson = created_lessons[0]

                    new_lesson = Lesson(
                        lesson_date=base_lesson.lesson_date,
                        time_frame=base_lesson.time_frame,
                        lesson_type=base_lesson.lesson_type,
                        group_priv=base_lesson.group_priv,
                        price_pl=base_lesson.price_pl,
                        client=real_client.full_name,
                        height_cm=rider.get("height_cm"),
                        weight_kg=rider.get("weight_kg"),
                        horse="",
                        attendance="Pending",
                        payment=None
                    )

                    db.session.add(new_lesson)
                    db.session.flush()

                    created_lessons.append(new_lesson)
                    continue



            except Exception as e:
                print("RIDER ERROR:", rider, e)   # 👈 add this  
                errors.append(str(e))
                continue

        # ---------------------------------------------------------
        # FINALIZE
        # ---------------------------------------------------------
        if created_lessons:
            # Always link invite to the FIRST lesson created/updated
            first_lesson = created_lessons[0]
            invite.lesson_id = first_lesson.lesson_id

            # Mark invite as completed
            invite.status = "completed"

            # Mark submission as processed
            sub.processed = True

        else:
            # Safety fallback — should never happen now
            sub.processed = True

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return f"DB commit failed: {e}"

        if errors:
            return f"Processed {len(created_lessons)} riders into lessons; errors: {len(errors)}"
        return f"Processed {len(created_lessons)} riders into lessons"



    @app.route('/notifications/invite_conflicts')
    def invite_conflicts():
        rows = (
            db.session.query(IncomingSubmission)
            .filter_by(needs_client_match=True, processed=False)
            .order_by(IncomingSubmission.id.asc())
            .all()
        )
        return render_template('invite_conflict_queue.html', rows=rows)


    @app.route('/notifications/invite_conflict/<int:submission_id>/<int:rider_index>')
    def invite_conflict_resolution(submission_id, rider_index):
        row = db.session.query(IncomingSubmission).get_or_404(submission_id)

        riders = parse_jotform_payload(
            row.raw_payload,
            forced_submission_id=row.id
        )

        # rider_index is 1‑based
        rider = riders[rider_index - 1]

        # Soft-match to get candidate matches
        rider_name = rider.get("full_name") or rider.get("name") or ""
        match_type, match_data = soft_match_client_for_invite(rider_name)

        if match_type == "exact":
            matches = [match_data]
        elif match_type == "none":
            matches = []
        else:
            matches = match_data  # list of clients

        return render_template(
            'conflict_resolution.html',
            submission=row,
            rider=rider,
            matches=matches,
            rider_index=rider_index,
            invite_mode=True  # tells template to hide overwrite option
        )

    @app.route('/notifications/invite_conflict/<int:submission_id>/<int:rider_index>', methods=['POST'])
    def finalize_invite_conflict(submission_id, rider_index):
        choice = request.form.get("choice")
        client_id = request.form.get("client_id")

        row = db.session.query(IncomingSubmission).get_or_404(submission_id)

        riders = parse_jotform_payload(
            row.raw_payload,
            forced_submission_id=row.id
        )
        rider = riders[rider_index - 1]

        rider_name = rider.get("full_name") or rider.get("name") or ""

        # ---------------------------------------------------------
        # RESOLVE CLIENT BASED ON CHOICE
        # ---------------------------------------------------------
        if choice == "use_existing" and client_id:
            client = db.session.query(Client).get(int(client_id))

        elif choice == "new":
            client = Client(
                full_name=rider_name,
                height_cm=rider.get("height_cm"),
                weight_kg=rider.get("weight_kg"),
                guardian_name="",
                mobile="",
                notes="Created via SMS invite"
            )
            db.session.add(client)
            db.session.flush()

        elif choice == "new_same_name":
            base = rider_name
            counter = 2
            while True:
                candidate = f"{base} ({counter})"
                exists = db.session.query(Client).filter_by(full_name=candidate).first()
                if not exists:
                    break
                counter += 1

            client = Client(
                full_name=candidate,
                height_cm=rider.get("height_cm"),
                weight_kg=rider.get("weight_kg"),
                guardian_name="",
                mobile="",
                notes="Created via SMS invite"
            )
            db.session.add(client)
            db.session.flush()

        elif choice == "ignore":
            row.processed = True
            row.ignored = True
            db.session.commit()
            return redirect(url_for('invite_conflicts'))

        else:
            return "Invalid choice", 400

        # ---------------------------------------------------------
        # STORE RESOLVED CLIENT ID ON RIDER OBJECT
        # ---------------------------------------------------------
        rider["resolved_client_id"] = client.client_id

        # ---------------------------------------------------------
        # CHECK IF ALL RIDERS ARE NOW RESOLVED
        # ---------------------------------------------------------
        all_resolved = True
        for r in riders:
            if not r.get("resolved_client_id"):
                all_resolved = False
                break

        if not all_resolved:
            db.session.commit()
            return redirect(url_for('invite_conflicts'))

        # ---------------------------------------------------------
        # ALL RIDERS RESOLVED → CLEAR FLAG AND REPROCESS
        # ---------------------------------------------------------
        row.needs_client_match = False
        db.session.commit()

        return redirect(url_for('process_all_pending'))


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
        clean = (client_key or "").strip().lower()

        # If no client key, return empty list immediately
        if not clean:
            return jsonify([])

        today = date.today()

        rows = (
            db.session.query(Lesson)
            .filter(
                Lesson.client.isnot(None),  # prevent None from breaking lower()
                func.replace(func.lower(Lesson.client), " ", "") == clean
            )
            .filter(Lesson.lesson_date < today)
            .order_by(Lesson.lesson_date.desc(), Lesson.lesson_id.desc())
            .limit(10)
            .all()
        )

        horses = [(r.horse or "").strip() for r in rows]

        return jsonify(horses)

    
    @app.route('/send_invite', methods=['POST'])
    def send_invite():
        print("DEBUG AUTH:", os.getenv("CLICKSEND_USERNAME"), os.getenv("CLICKSEND_API_KEY"))
        print("SMS INVITE POST:", dict(request.form))
        # ------------------------------
        # 1. Extract posted fields
        # ------------------------------
        lesson_date_str = (
            request.form.get("invite_date")
            or request.form.get("lesson_date")
            or request.form.get("date")
        )

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

        mobile_raw = (
            request.form.get("invite_mobile")
            or request.form.get("mobile")
            or request.form.get("client_phone")
            or ""
        )

        riders_requested_raw = (
            request.form.get("invite_riders")
            or request.form.get("riders_requested")
            or "1"
        )

        cost_per_person_raw = (
            request.form.get("invite_cost")
            or request.form.get("cost_per_person")
            or "0"
        )

        teacher_id_raw = request.form.get("invite_teacher_id")
        try:
            teacher_id = int(teacher_id_raw)
        except:
            teacher_id = None

        # Canonical time frame
        lesson_time_str = invite_time_frame

        # ------------------------------
        # 2. Validate required fields
        # ------------------------------
        digits = re.sub(r'\D+', '', mobile_raw or "")
        if digits.startswith("0"):
            digits = digits[1:]
        mobile_clean = "+61" + digits if digits else ""

        if not lesson_date_str or not lesson_time_str or not mobile_clean:
            flash("Missing lesson date, time, or mobile number for invite.", "danger")
            return redirect(url_for('lessons_by_date', date=lesson_date_str or date.today().isoformat()))

        try:
            riders_requested = int(riders_requested_raw)
        except:
            riders_requested = 1

        try:
            cost_per_person = float(cost_per_person_raw)
        except:
            cost_per_person = 0.0

        # ------------------------------
        # 3. Generate token
        # ------------------------------
        token = generate_invite_token(lesson_date_str, lesson_time_str)

        # ------------------------------
        # 4. CREATE PLACEHOLDER CLIENT
        # ------------------------------
        placeholder_client = Client(
            full_name=f"[Pending {token}]",
            mobile=mobile_clean,
            notes="Pending SMS invite"
        )
        db.session.add(placeholder_client)
        db.session.flush()

        # ------------------------------
        # 5. CREATE PENDING LESSON
        # ------------------------------
        lesson = Lesson(
            lesson_date=datetime.strptime(lesson_date_str, "%Y-%m-%d").date(),
            time_frame=lesson_time_str,
            lesson_type=invite_lesson_type,
            group_priv=invite_group_priv,
            price_pl=cost_per_person,
            client=placeholder_client.full_name,
            horse=""
        )
        db.session.add(lesson)
        db.session.flush()

        # ------------------------------
        # 6. CREATE LESSON INVITE
        # ------------------------------
        invite = LessonInvite(
            lesson_id=lesson.lesson_id,
            token=token,
            mobile=mobile_clean,
            riders_requested=riders_requested,
            cost_per_person=cost_per_person,
            time_frame=lesson_time_str,
            lesson_type=invite_lesson_type,
            group_priv=invite_group_priv,
            status="awaiting_form",
            lesson_date=datetime.strptime(lesson_date_str, "%Y-%m-%d").date()
        )

        db.session.add(invite)
        db.session.commit()

        # ------------------------------
        # 7. Build SMS + send
        # ------------------------------
        base_url = "https://form.jotform.com/253599154628066"
        params = urlencode({
            "i_t": token,
            "r_r": riders_requested
        })
        jotform_url = f"{base_url}?{params}"

        from dateutil import parser
        try:
            date_obj = parser.parse(lesson_date_str)
            formatted_date = date_obj.strftime("%-d %b")
        except:
            formatted_date = lesson_date_str

        m = re.search(r'(\d{1,2}:\d{2})', lesson_time_str or "")
        start_24 = m.group(1) if m else lesson_time_str.split("-")[0].strip()

        try:
            t_obj = datetime.strptime(start_24, "%H:%M")
            hour_12 = t_obj.strftime("%I").lstrip("0")
            minute = t_obj.strftime("%M")
            ampm = t_obj.strftime("%p").lower()
            start_pretty = f"{hour_12}.{minute}{ampm}"
        except:
            start_pretty = start_24

        rider_word = "rider" if riders_requested == 1 else "riders"

        sms_text = (
            f"Hi! Please confirm your lesson on {formatted_date} at {start_pretty}. "
            f"{riders_requested} {rider_word}, ${cost_per_person:.0f} ea. "
            f"{jotform_url}"
        )

        sender = app.config.get("EQUESTRIAN_SENDER", "")

        ok = send_sms_clicksend(mobile_clean, sms_text, sender)
        flash("SMS invite sent successfully." if ok else "Failed to send SMS invite.", 
              "success" if ok else "danger")

        return redirect(url_for('lessons_by_date', date=lesson_date_str))



    # ---------------- ROUTES ---------------- #

    @app.post("/save_booking")
    def save_booking():
        data = request.get_json()

        booking_type = data.get("booking_type")

        if booking_type == "existing":
            return handle_existing_client(data)

        if booking_type == "new":
            return handle_new_client(data)

        return jsonify(success=False, error="Invalid booking type")




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
        """
        Fast notifications listing:
        - Limits rows rendered (prevents timeouts)
        - Parses payload in LIGHT mode (no client-table load / no matching)
        """
        page = request.args.get("page", default=1, type=int)
        per_page = request.args.get("per_page", default=50, type=int)
        per_page = max(10, min(per_page, 200))  # clamp

        rows = (
            db.session.query(IncomingSubmission)
            .filter_by(processed=False)
            .order_by(IncomingSubmission.received_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        for r in rows:
            try:
                riders = parse_jotform_payload(
                    r.raw_payload,
                    forced_submission_id=r.form_id,
                    clients_cache=None,
                    mode="light"
                )

                names = []
                for rider in riders:
                    n = rider.get("name")
                    if n:
                        names.append(n)

                r.display_names = ", ".join(names) if names else "Invite Submission"

            except Exception as e:
                print("ERROR extracting names:", e)
                r.display_names = "(unknown)"

        return render_template(
            'notifications.html',
            rows=rows,
            page=page,
            per_page=per_page
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
                unique_hash=payload_hash
            ).first()

            if existing_row:
                continue

            # Insert into incoming_submissions
            row = IncomingSubmission(
                form_id=FORM_ID,
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
        riders = parse_jotform_payload(
            submission.raw_payload,
            forced_submission_id=submission.form_id
        )

        # NEW: filter valid riders (skip incomplete)
        valid_riders = [r for r in riders if not r.get("incomplete")]

        # NEW: if all riders incomplete → auto-ignore
        if not valid_riders:
            submission.processed = True
            submission.ignored = True
            submission.processed_at = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('notifications'))

        # NEW: conflict detection only on valid riders
        for i, rider in enumerate(valid_riders, start=1):
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
            clients=valid_riders
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

        # PHASE 3: Parse riders ONCE
        riders = parse_jotform_payload(
            row.raw_payload,
            forced_submission_id=row.form_id
        )
        rider = riders[rider_index - 1]

        # Skip incomplete riders (safety)
        if rider.get("incomplete"):
            row.processed = True
            row.processed_at = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # NEW: ignore option
        if choice == "ignore":
            row.processed = True
            row.ignored = True
            row.processed_at = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # PHASE 3: Preload clients ONCE
        all_clients = db.session.query(Client).all()
        clients_by_id = {c.id: c for c in all_clients}

        existing = clients_by_id.get(int(client_id)) if client_id else None

        # Extract fields
        name = normalise_full_name(rider["name"])
        age = int(rider["age"]) if rider["age"] else None
        guardian = rider["guardian"]
        mobile = clean_mobile(rider["mobile"])
        email = rider["email"]
        disclaimer = int(rider["disclaimer"]) if rider["disclaimer"] else None
        jotform_id = str(row.form_id)

        # -----------------------------------------
        # USE EXISTING (FAST)
        # -----------------------------------------
        if choice == "use_existing" and existing:
            existing.jotform_submission_id = jotform_id
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # OVERWRITE EXISTING (FAST)
        # -----------------------------------------
        if choice == "overwrite" and existing:
            existing.full_name = name
            existing.age = age
            existing.guardian_name = guardian
            existing.mobile = mobile
            existing.email_primary = email
            existing.disclaimer = disclaimer
            existing.height_cm = rider.get("height_cm")
            existing.weight_kg = rider.get("weight_kg")
            existing.notes = rider.get("notes")
            existing.jotform_submission_id = jotform_id
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # CREATE NEW CLIENT (FAST)
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
        # CREATE NEW CLIENT (SAME NAME) (FAST)
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

        # PHASE 3: Parse riders ONCE
        riders = parse_jotform_payload(
            row.raw_payload,
            forced_submission_id=row.form_id
        )

        jotform_id = str(row.form_id)

        # PHASE 3: Skip incomplete riders
        valid_riders = [r for r in riders if not r.get("incomplete")]

        # If all riders incomplete → auto-ignore
        if not valid_riders:
            row.processed = True
            row.ignored = True
            row.processed_at = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('notifications'))

        # PHASE 3: Preload clients ONCE
        all_clients = db.session.query(Client).all()
        clients_by_id = {c.id: c for c in all_clients}

        for i, rider in enumerate(valid_riders, start=1):
            choice = request.form.get(f"client_choice_{i}")

            # Safety skip
            if rider.get("incomplete"):
                continue

            # Extract fields
            name = request.form.get(f"name_{i}") or rider.get("name") or "Unknown"
            age = request.form.get(f"age_{i}") or rider.get("age")
            guardian = request.form.get(f"guardian_{i}") or rider.get("guardian")

            raw_mobile = request.form.get(f"mobile_{i}") or rider.get("mobile")
            mobile = clean_mobile(raw_mobile) if raw_mobile else None

            email = request.form.get(f"email_{i}") or rider.get("email")

            raw_disclaimer = request.form.get(f"disclaimer_{i}") or rider.get("disclaimer")
            try:
                disclaimer = int(raw_disclaimer) if raw_disclaimer else None
            except ValueError:
                disclaimer = None

            height_cm = extract_number(request.form.get(f"height_{i}") or rider.get("height_cm"))
            weight_kg = extract_number(request.form.get(f"weight_{i}") or rider.get("weight_kg"))
            notes = request.form.get(f"notes_{i}") or rider.get("notes") or ""

            # --- CREATE NEW ---
            if choice == "new":
                new_client = Client(
                    full_name=name,
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
            if choice and choice.startswith("existing_"):
                existing_id = int(choice.replace("existing_", ""))
                client = clients_by_id.get(existing_id)
                if client:
                    client.jotform_submission_id = jotform_id
                    log_submission_link("USE_EXISTING", client, jotform_id)
                continue

            # --- OVERWRITE EXISTING ---
            if choice and choice.startswith("overwrite_"):
                existing_id = int(choice.replace("overwrite_", ""))
                client = clients_by_id.get(existing_id)
                if client:
                    client.full_name = name
                    client.guardian_name = guardian
                    client.age = age
                    client.mobile = mobile
                    client.email_primary = email
                    client.disclaimer = disclaimer
                    client.height_cm = height_cm
                    client.weight_kg = weight_kg
                    client.notes = notes
                    client.jotform_submission_id = jotform_id
                    log_submission_link("OVERWRITE_EXISTING", client, jotform_id)
                continue

        # SAFE LOGGING
        riders_for_log = parse_jotform_payload(
            row.raw_payload,
            forced_submission_id=row.form_id
        )
        safe_names = [r.get("name") for r in riders_for_log if r.get("name")]
        if safe_names:
            log_disclaimer_processed(safe_names)
        else:
            log_disclaimer_processed(["(invite submission)"])

        row.processed = True
        row.processed_at = datetime.utcnow()
        db.session.commit()

        return redirect(url_for('notifications'))

    @app.route('/notifications/process_all')
    def process_all_pending():
        from datetime import datetime

        # Get the next unprocessed + not ignored submission
        next_row = (
            db.session.query(IncomingSubmission)
            .filter_by(processed=False, ignored=False)
            .first()
        )

        if not next_row:
            return redirect(url_for('notifications'))

        # -----------------------------------------
        # PHASE 3: Parse riders ONCE
        # -----------------------------------------
        riders = parse_jotform_payload(
            next_row.raw_payload,
            forced_submission_id=next_row.form_id
        )

        # -----------------------------------------
        # PHASE 3: Skip incomplete riders
        # -----------------------------------------
        valid_riders = [r for r in riders if not r.get("incomplete")]

        # If all riders incomplete → auto-ignore
        if not valid_riders:
            next_row.processed = True
            next_row.ignored = True
            next_row.processed_at = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('process_all_pending'))

        # -----------------------------------------
        # PHASE 3: Preload clients ONCE
        # -----------------------------------------
        all_clients = db.session.query(Client).all()
        clients_by_name = {
            normalise_full_name(c.full_name): c
            for c in all_clients
        }

        # -----------------------------------------
        # PHASE 3: Conflict detection (fast)
        # -----------------------------------------
        for idx, rider in enumerate(valid_riders, start=1):
            matches = rider.get("matches", [])
            if matches:
                return redirect(url_for(
                    'finalize_conflict',
                    submission_id=next_row.id,
                    rider_index=idx
                ))

        # -----------------------------------------
        # PHASE 3: No conflicts → fast create
        # -----------------------------------------
        for rider in valid_riders:
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
        # Mark processed
        # -----------------------------------------
        names = [r["name"] for r in valid_riders]
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
        from sqlalchemy import func

        INVITE_FORM_ID = "253599154628066"
        API_KEY = os.getenv("JOTFORM_API_KEY", "")

        # Pull ALL submissions from JotForm for this form
        url = f"https://api.jotform.com/form/{INVITE_FORM_ID}/submissions?apiKey={API_KEY}"
        r = requests.get(url)
        data = r.json()

        submissions = data.get("content", [])
        count = 0

        for s in submissions:
            submission_id = s.get("id")  # UNIQUE PER SUBMISSION
            form_id = s.get("form_id")

            # Check if THIS EXACT submission is already stored
            existing = (
                db.session.query(IncomingSubmission)
                .filter(IncomingSubmission.submission_id == submission_id)
                .first()
            )
            if existing:
                continue

            # Store raw payload
            raw_payload = json.dumps(s)

            new_sub = IncomingSubmission(
                submission_id=submission_id,   # <-- MUST EXIST IN MODEL
                form_id=form_id,               # JotForm form ID (same for all)
                raw_payload=raw_payload,
                received_at=datetime.utcnow(),
                processed=False,
                needs_client_match=False
            )

            db.session.add(new_sub)
            db.session.commit()

            # Attempt immediate invite match
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
            if invite.status != "process":
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


    @app.post("/recalculate_client")
    def recalculate_client():
        client = request.form.get("client")

        # get all lessons for this client
        lessons = Lesson.query.filter_by(client=client).all()

        for l in lessons:
            carry = l.carry_fwd or 0
            payment = l.payment or 0
            price = l.price_pl or 0
            att = (l.attendance or '').strip().upper()

            balance = carry + payment
            if att in ['Y', 'N']:
                balance -= price

            l.balance = balance

        db.session.commit()

        flash(f"Lessons recalculated for client: {client}", "success")
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