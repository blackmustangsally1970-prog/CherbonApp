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


def run_import():
    app = create_app()
    with app.app_context():
        print("Client height/weight updater started…")

        with open("clients_update.csv", newline='', encoding="utf-8", errors="replace") as f:

            reader = csv.DictReader(f)

            print("HEADERS:", reader.fieldnames)

            updated = 0
            skipped = 0

            for row in reader:
                full_name = row.get("full_name", "").strip()

                if not full_name:
                    print("⚠️  Skipped row with no full_name")
                    skipped += 1
                    continue

                client = Client.query.filter_by(full_name=full_name).first()

                if not client:
                    print(f"❌ No match for: {full_name} — SKIPPED")
                    skipped += 1
                    continue

                # Parse values
                new_weight = parse_int(row.get("weight_kg"))
                new_height = parse_int(row.get("height_cm"))

                changed = False

                if new_weight is not None:
                    client.weight_kg = new_weight
                    changed = True

                if new_height is not None:
                    client.height_cm = new_height
                    changed = True

                if changed:
                    try:
                        db.session.commit()
                        print(f"✔ Updated: {full_name}  (W:{new_weight}, H:{new_height})")
                        updated += 1
                    except Exception as e:
                        print(f"❌ Commit failed for {full_name}: {e}")
                        db.session.rollback()
                        skipped += 1


            print("\n=== IMPORT COMPLETE ===")
            print(f"Updated: {updated}")
            print(f"Skipped: {skipped}")


# ⭐ THIS WAS MISSING — NOW THE SCRIPT ACTUALLY RUNS
if __name__ == "__main__":
    run_import()
