from flask import Blueprint, redirect, url_for, session
from datetime import datetime
from extensions import db
from permissions import is_admin, is_coordinator, is_caterer
from weddings.models import JobListInstance

job_instance_bp = Blueprint('job_instance', __name__, url_prefix='/weddings')


# ---------------------------------------------------------
# START A JOB
# ---------------------------------------------------------
@job_instance_bp.route('/<int:wedding_id>/jobs/start/<int:job_id>/<int:employee_id>')
def job_start(wedding_id, job_id, employee_id):

    # Caterers cannot use job engine
    if is_caterer():
        return "Unauthorized", 403

    job = JobListInstance.query.get_or_404(job_id)

    # If someone else is already doing it, ignore
    if job.in_progress_by and job.in_progress_by != employee_id:
        return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

    # Staff can only start their own jobs
    if session.get('role') not in ['admin', 'management', 'coordinator']:
        if employee_id != session.get('employee_id'):
            return "Unauthorized", 403

    job.in_progress_by = employee_id
    job.started_at = datetime.utcnow()

    db.session.commit()
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# FINISH A JOB
# ---------------------------------------------------------
@job_instance_bp.route('/<int:wedding_id>/jobs/finish/<int:job_id>/<int:employee_id>')
def job_finish(wedding_id, job_id, employee_id):

    # Caterers cannot use job engine
    if is_caterer():
        return "Unauthorized", 403

    job = JobListInstance.query.get_or_404(job_id)

    # Admin / Management can finish any job
    if is_admin():
        pass

    # Coordinator can finish ANY job (supervisor)
    elif is_coordinator():
        pass

    # Staff can only finish jobs they started
    else:
        if job.in_progress_by != employee_id:
            return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))

    job.in_progress_by = None
    job.completed = True
    job.completed_by = employee_id
    job.completed_at = datetime.utcnow()

    db.session.commit()
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))
