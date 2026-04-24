import os
os.environ["DISABLE_WEASYPRINT"] = "1"
os.environ["DISABLE_PLAYWRIGHT"] = "1"

import sys
import csv

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from extensions import db
from models import Client


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_mobile(value):
    if not value:
        return None
    digits = "".join(c for c in value if c.isdigit())
    if digits.startswith("61") and len(digits) > 9:
        digits = "0" + digits[2:]
    if len(digits) == 9:
        digits = "0" + digits
    return digits


def build_client(row):
    return Client(
        full_name=row.get("full_name", "").strip(),
        guardian_name=row.get("guardian1", "").strip(),
        age=parse_int(row.get("guardian1_age")),
        disclaimer=parse_int(row.get("disclaimer")),
        notes=row.get("notes", "").strip(),
        email_primary=row.get("email_prin", "").strip(),
        email_secondary=row.get("email_sec", "").strip(),
        email_guardian2=row.get("email_guardian2", "").strip(),
        mobile=clean_mobile(row.get("mobile", "")),
        mobile2=clean_mobile(row.get("mobile2", "")),
        ndis_number=row.get("guardian2_ndis_num", "").strip(),
        ndis_code=row.get("ndis_code", "").strip(),
        notes2=row.get("notes2", "").strip(),
        jotform_submission_id=row.get("jotform_submission_i", "").strip(),
        weight_kg=parse_int(row.get("weight_kg")),
        height_cm=parse_int(row.get("height_cm")),
    )


def run_import():
    app = create_app()
    with app.app_context():
        print("Importer started…")

        with open("clients.csv", newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)

            print("HEADERS:", reader.fieldnames)

            for row in reader:
                client = build_client(row)
                db.session.add(client)
                print(f"Queued: {client.full_name}")

            db.session.commit()
            print("Import complete.")


if __name__ == "__main__":
    run_import()
