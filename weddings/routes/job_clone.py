from flask import Blueprint, redirect, url_for, flash
from extensions import db
from weddings.models import JobListMaster, JobListInstance

job_clone_bp = Blueprint('job_clone', __name__, url_prefix='/weddings')

# ---------------------------------------------------------
# CLONE MASTER JOB LIST INTO WEDDING
# ---------------------------------------------------------
@job_clone_bp.route('/<int:wedding_id>/jobs/clone_master')
def clone_master_jobs(wedding_id):

    masters = JobListMaster.query.filter_by(active=True).order_by(
        JobListMaster.role.asc(),
        JobListMaster.sort_order.asc()
    ).all()

    if not masters:
        flash("No active master jobs to insert.", "warning")
        return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

    for m in masters:
        new_job = JobListInstance(
            wedding_id=wedding_id,
            master_id=m.id,
            description=m.description,
            employee_id=None,        # optional owner
            in_progress_by=None,
            completed=False,
            completed_by=None,
            started_at=None,
            completed_at=None
        )
        db.session.add(new_job)

    db.session.commit()

    flash("Master job list inserted.", "success")
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))
