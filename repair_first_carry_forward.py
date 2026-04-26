# repair_first_carry_forward.py
# One-off script to restore FIRST lesson carry_fwd per client
# using the original lessons CSV.

import csv
from datetime import datetime
from app import app, db
from models import Lesson

CSV_FILE = "import_tools/lessons.csv"   # <-- your lessons CSV

# Helper to parse dates safely
def parse_date(s):
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except:
        return None

# Temporary structure:
# first_lessons[client] = { "date": date, "carry": carry_fwd }
first_lessons = {}

with app.app_context():
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            client = (row.get("client") or row.get("full_name") or "").strip()
            if not client:
                continue

            date_str = row.get("lesson_date") or row.get("date")
            date = parse_date(date_str)
            if not date:
                continue

            raw_carry = row.get("carry_fwd") or row.get("carry") or "0"
            try:
                carry = float(raw_carry)
            except:
                carry = 0.0

            # If this is the first time we see this client OR this date is earlier
            if client not in first_lessons or date < first_lessons[client]["date"]:
                first_lessons[client] = {
                    "date": date,
                    "carry": carry
                }

    # Now update DB
    updated = 0
    missing = []

    for client, info in first_lessons.items():
        date = info["date"]
        carry = info["carry"]

        # Find the earliest lesson for this client in DB
        lesson = (
            Lesson.query
            .filter_by(client=client)
            .order_by(Lesson.lesson_date.asc(), Lesson.lesson_id.asc())
            .first()
        )

        if not lesson:
            missing.append(client)
            continue

        # Update ONLY the first lesson's carry_fwd
        lesson.carry_fwd = carry
        updated += 1

    db.session.commit()

print(f"Updated first carry_fwd for {updated} clients.")
print(f"Clients missing in DB: {len(missing)}")
for m in missing:
    print(" -", m)
