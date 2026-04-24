# repair_client_ages.py
# One-off script to update client ages from /import_tools/clients.csv

import csv
from app import db
from models import Client   # <-- correct model name

CSV_FILE = "import_tools/clients.csv"   # <-- uses your existing file

updated = 0
skipped = 0
not_found = []

with open(CSV_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        name = row.get("full_name") or row.get("client") or ""
        raw_age = row.get("age")

        # Skip rows with no name
        if not name:
            skipped += 1
            continue

        # Skip rows with no age
        if raw_age in [None, "", " ", "None"]:
            skipped += 1
            continue

        # Convert age safely
        try:
            age = int(float(raw_age))
        except:
            skipped += 1
            continue

        # Exact match lookup
        client = Client.query.filter_by(full_name=name).first()

        if not client:
            not_found.append(name)
            skipped += 1
            continue

        # Update ONLY age
        client.age = age
        updated += 1

db.session.commit()

print(f"Updated ages for {updated} clients.")
print(f"Skipped {skipped} rows.")
print(f"Clients not found: {len(not_found)}")

if not_found:
    print("\nMissing clients:")
    for n in not_found:
        print(" -", n)
