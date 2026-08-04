from extensions import db
from cherbonapp.models import Account, LegacyAccountMap
from models import Users, Employee, Caterer   # your legacy models

def migrate_accounts():
    print("Starting migration...")

    # 1. Migrate USERS → Account
    users = Users.query.all()
    for u in users:
        acc = Account(
            full_name=u.full_name or u.username,
            username=u.username,
            password_hash=u.password_hash,
            primary_role="admin",
            is_admin=True,
            active=True
        )
        db.session.add(acc)
        db.session.flush()

        db.session.add(LegacyAccountMap(
            legacy_table="users",
            legacy_id=u.id,
            account_id=acc.id
        ))

    # 2. Migrate EMPLOYEES → Account
    employees = Employee.query.all()
    for e in employees:
        acc = Account(
            full_name=e.full_name,
            phone=e.phone,
            setup_code=e.setup_code,
            pin_hash=e.pin_hash,
            primary_role="staff",
            is_staff=True,
            active=True
        )
        db.session.add(acc)
        db.session.flush()

        db.session.add(LegacyAccountMap(
            legacy_table="employees",
            legacy_id=e.id,
            account_id=acc.id
        ))

    # 3. Migrate CATERER → Account
    caterers = Caterer.query.all()
    for c in caterers:
        acc = Account(
            full_name=c.contact_name or c.login_email,
            username=c.login_email,
            password_hash=c.password_hash,
            phone=c.phone,
            notes=c.notes,
            primary_role="caterer",
            is_caterer=True,
            active=True
        )
        db.session.add(acc)
        db.session.flush()

        db.session.add(LegacyAccountMap(
            legacy_table="caterer",
            legacy_id=c.id,
            account_id=acc.id
        ))

    db.session.commit()
    print("Migration complete.")
