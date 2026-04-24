import os
os.environ["DISABLE_WEASYPRINT"] = "1"
os.environ["DISABLE_PLAYWRIGHT"] = "1"

import sys
import csv
from datetime import datetime

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from extensions import db
from models import Lesson


def parse_date(value):
    if not value or str(value).strip() == "":
        return None

    value = str(value).strip()

    # Try dd/mm/yyyy
    try:
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError:
        pass

    # Try yyyy-mm-dd
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        pass

    # Try dd-mm-yyyy (some exports do this)
    try:
        return datetime.strptime(value, "%d-%m-%Y").date()
    except ValueError:
        pass

    # If everything fails, return None instead of exploding
    return None


def parse_float(value):
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def build_lesson(row):
    return Lesson(
        # lesson_id comes from DB, ignore CSV lesson_id
        lesson_date=parse_date(row.get("lesson_date")),
        lesson_type=row.get("lesson_type", "").strip(),
        client=row.get("client", "").strip(),
        time_frame=row.get("time_frame", "").strip(),
        blockends=parse_date(row.get("blockends")),
        attendance=row.get("attendance", "").strip(),
        payment=parse_float(row.get("payment")),
        price_pl=parse_float(row.get("price_pl")),
        carry_fwd=parse_float(row.get("carry_fwd")),
        balance=parse_float(row.get("balance")),
        horse=row.get("horse", "").strip(),
        group_priv=row.get("group_priv", "").strip(),
        lesson_no=row.get("lesson_no", "").strip(),
        freq=row.get("freq", "").strip(),
        adjust=parse_int(row.get("adjust")),
        block_key=row.get("block_key", "").strip(),
        # lesson_no1 exists in CSV but no field in model → ignored
        # lesson_notes not in CSV → stays NULL
    )


def run_import():
    app = create_app()
    with app.app_context():
        print("Lesson importer started…")

        csv_path = os.path.join(os.path.dirname(__file__), "lessons.csv")
        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print("HEADERS:", reader.fieldnames)

            count = 0
            for row in reader:
                lesson = build_lesson(row)
                db.session.add(lesson)
                count += 1
                if count % 100 == 0:
                    print(f"Queued {count} lessons…")

            db.session.commit()
            print(f"Import complete. Total lessons imported: {count}")


if __name__ == "__main__":
    run_import()
