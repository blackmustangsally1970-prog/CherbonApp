# repair_first_carry_forward.py
# Correct version: updates the DB lesson whose date matches the CSV's first lesson date per client.

import csv
from datetime import datetime
from app import app, db
from models import Lesson

CSV_FILE = "import_tools/lessons.csv"

def clean(s):
    if not s:
        return ""
    return s.replace("\ufeff", "").strip()

def parse_date(s):
    if not s:
        return None
    s = clean(s)
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    return None

def parse_float(s):
    try:
        return float(clean(s))
    except:
        return 0.0

with app.app_context():

    # STEP 1 — Read CSV and determine the FIRST lesson per client
    first_lessons = {}

    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Clean header names
        reader.fieldnames = [clean(h) for h in reader.fieldnames]

        for row in reader:
            row = {clean(k): clean(v) for k, v in row.items()}

            client = row.get("client")
            if not client:
                continue

            date = parse_date(row.get("lesson_date"))
            if not date:
                continue

            carry = parse_float(row.get("carry_fwd"))

            # Track earliest CSV lesson per client
            if client not in first_lessons or date < first_lessons[client]["date"]:
                first_lessons[client] = {"date": date, "carry": carry}

    # STEP 2 — Update DB lesson that matches the CSV's first lesson date
    updated = 0
    missing = []
    no_date_match = []

    for client, info in first_lessons.items():
        csv_date = info["date"]
        carry = info["carry"]

        # Find DB lesson with SAME DATE and SAME CLIENT
        lesson = (
            Lesson.query
            .filter_by(client=client, lesson_date=csv_date)
            .order_by(Lesson.lesson_id.asc())
            .first()
        )

        if not lesson:
            no_date_match.append((client, csv_date))
            continue

        lesson.carry_fwd = carry
        updated += 1

    db.session.commit()

print(f"Updated carry_fwd for {updated} clients.")
print(f"No DB lesson found for {len(no_date_match)} clients with matching date.")

if no_date_match:
    print("\nClients with no matching DB date:")
    for client, date in no_date_match:
        print(f" - {client} (CSV first date: {date})")
