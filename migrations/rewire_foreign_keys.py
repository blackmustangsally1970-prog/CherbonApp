from extensions import db
from models import LegacyAccountMap, Account
from weddings.models.wedding_staff_assignment import WeddingStaffAssignment
from weddings.models.job_list_instance import JobListInstance
from weddings.models.wedding import Wedding


def map_id(legacy_table, legacy_id):
    """Return the new account_id for a legacy row."""
    mapping = LegacyAccountMap.query.filter_by(
        legacy_table=legacy_table,
        legacy_id=legacy_id
    ).first()

    if not mapping:
        print(f"WARNING: No mapping found for {legacy_table} id={legacy_id}")
        return None

    return mapping.account_id


def rewire_foreign_keys():
    print("Starting foreign key rewiring...")

    # 1. WeddingStaffAssignment
    assignments = WeddingStaffAssignment.query.all()
    for a in assignments:
        new_id = map_id("employees", a.employee_id)
        if new_id:
            a.account_id = new_id

    # 2. JobListInstance
    jobs = JobListInstance.query.all()
    for j in jobs:
        if j.employee_id:
            j.account_id = map_id("employees", j.employee_id)

        if j.in_progress_by:
            j.in_progress_by = map_id("employees", j.in_progress_by)

        if j.completed_by:
            j.completed_by = map_id("employees", j.completed_by)

    # 3. Wedding (caterer)
    weddings = Wedding.query.all()
    for w in weddings:
        if w.caterer_id:
            w.caterer_id = map_id("caterer", w.caterer_id)

    db.session.commit()
    print("Foreign key rewiring complete.")
