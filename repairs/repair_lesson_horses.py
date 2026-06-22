# repair_lesson_horses.py
# One-off script to update lesson.horse from /import_tools/lessons.csv

import csv
from datetime import datetime
from app import app, db
from models import Lesson

CSV_FILE = "import_tools/lessons.csv"

updated = 0
skipped = 0
not_found = []

def convert_date(d):
    """Convert DD/MM/YYYY → YYYY-MM-DD safely."""
    d = d.strip()
    if "/" in d:
        try:
            return datetime.strptime(d, "%d/%m/%Y").date()
        except:
            return None
    return d  # already correct or empty


with app.app_context():
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            client = row.get("client") or row.get("client_name") or ""
            raw_date = row.get("lesson_date") or row.get("date") or ""
            time_frame = row.get("time_frame") or row.get("time") or ""
            horse = row.get("horse") or ""

            # Skip rows missing identifiers
            if not client or not raw_date or not time_frame:
                skipped += 1
                continue

            # Convert date
            date = convert_date(raw_date)
            if not date:
                skipped += 1
                continue

            # Skip blank horse
            if horse.strip() == "":
                skipped += 1
                continue

            # Find the exact lesson
            lesson = Lesson.query.filter_by(
                client=client,
                lesson_date=date,
                time_frame=time_frame
            ).first()

            if not lesson:
                not_found.append(f"{client} @ {raw_date} {time_frame}")
                skipped += 1
                continue

            # Update ONLY the horse field
            lesson.horse = horse.strip()
            updated += 1

    db.session.commit()

print(f"Updated horse for {updated} lessons.")
print(f"Skipped {skipped} rows.")
print(f"Lessons not found: {len(not_found)}")

if not_found:
    print("\nMissing lessons:")
    for n in not_found:
        print(" -", n)
