# repair_client_emails.py
# One-off script to update client email_primary from /import_tools/clients.csv

import csv
from app import app, db
from models import Client

CSV_FILE = "import_tools/clients.csv"

updated = 0
skipped = 0
not_found = []

POSSIBLE_EMAIL_HEADERS = [
    "email_primary",
    "email_prin",
    "email",
    "email_primary ",
    "email_primary\r",
    "email_primary﻿",   # BOM
]

with app.app_context():
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = (row.get("full_name") or row.get("client") or "").strip()

            if not name:
                skipped += 1
                continue

            # Find the email column safely
            raw_email = None
            for key in POSSIBLE_EMAIL_HEADERS:
                if key in row and row[key] not in [None, "", " ", "None"]:
                    raw_email = row[key].strip()
                    break

            if not raw_email:
                skipped += 1
                continue

            # Lookup client
            client = Client.query.filter_by(full_name=name).first()

            if not client:
                not_found.append(name)
                skipped += 1
                continue

            # Update ONLY email_primary
            client.email_primary = raw_email
            updated += 1

    db.session.commit()

print(f"Updated email_primary for {updated} clients.")
print(f"Skipped {skipped} rows.")
print(f"Clients not found: {len(not_found)}")

if not_found:
    print("\nMissing clients:")
    for n in not_found:
        print(" -", n)
