from flask import Blueprint, redirect, url_for, session
from datetime import datetime
from app import db
from weddings.models import JobListInstance

job_instance_bp = Blueprint('job_instance', __name__, url_prefix='/weddings')

# ---------------------------------------------------------
# START A JOB
# ---------------------------------------------------------
@job_instance_bp.route('/<int:wedding_id>/jobs/start/<int:job_id>/<int:employee_id>')
def job_start(wedding_id, job_id, employee_id):
    job = JobListInstance.query.get_or_404(job_id)

    # If someone else is already doing it, ignore
    if job.in_progress_by and job.in_progress_by != employee_id:
        return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

    job.in_progress_by = employee_id
    job.started_at = datetime.utcnow()

    db.session.commit()
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# FINISH A JOB
# ---------------------------------------------------------
@job_instance_bp.route('/<int:wedding_id>/jobs/finish/<int:job_id>/<int:employee_id>')
def job_finish(wedding_id, job_id, employee_id):
    job = JobListInstance.query.get_or_404(job_id)

    # Only the person doing it OR coordinator can finish
    if job.in_progress_by != employee_id and session.get('role') != 'coordinator':
        return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

    job.in_progress_by = None
    job.completed = True
    job.completed_by = employee_id
    job.completed_at = datetime.utcnow()

    db.session.commit()
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))
