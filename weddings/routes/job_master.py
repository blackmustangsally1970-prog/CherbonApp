from flask import Blueprint, render_template, request, redirect, url_for, flash
from extensions import db
from auth.permissions import is_admin, is_coordinator, is_caterer
from weddings.models import JobListMaster

job_master_bp = Blueprint('job_master', __name__, url_prefix='/weddings')


# ---------------------------------------------------------
# MASTER JOB LIST PAGE
# Admin / Management ONLY
# ---------------------------------------------------------
@job_master_bp.route('/job_master')
def job_master():

    # Coordinators cannot manage master templates
    if is_coordinator():
        return "Unauthorized", 403

    # Caterers cannot access job engine
    if is_caterer():
        return "Unauthorized", 403

    # Admin + Management only
    if not is_admin():
        return "Unauthorized", 403

    jobs = JobListMaster.query.order_by(JobListMaster.sort_order.asc()).all()
    return render_template('weddings/job_master.html', jobs=jobs)


# ---------------------------------------------------------
# ADD MASTER JOB
# Admin / Management ONLY
# ---------------------------------------------------------
@job_master_bp.route('/job_master/add', methods=['POST'])
def job_master_add():

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    role = request.form.get('role')
    description = request.form.get('description')

    if not description:
        flash("Description required", "danger")
        return redirect(url_for('job_master.job_master'))

    max_order = db.session.query(db.func.max(JobListMaster.sort_order)).scalar() or 0

    new_job = JobListMaster(
        role=role,
        description=description,
        sort_order=max_order + 1
    )

    db.session.add(new_job)
    db.session.commit()

    return redirect(url_for('job_master.job_master'))


# ---------------------------------------------------------
# DELETE MASTER JOB
# Admin / Management ONLY
# ---------------------------------------------------------
@job_master_bp.route('/job_master/delete/<int:job_id>')
def job_master_delete(job_id):

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    job = JobListMaster.query.get_or_404(job_id)
    db.session.delete(job)
    db.session.commit()

    return redirect(url_for('job_master.job_master'))


# ---------------------------------------------------------
# UPDATE MASTER JOB
# Admin / Management ONLY
# ---------------------------------------------------------
@job_master_bp.route('/job_master/update/<int:job_id>', methods=['POST'])
def job_master_update(job_id):

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    job = JobListMaster.query.get_or_404(job_id)

    job.role = request.form.get('role')
    job.description = request.form.get('description')
    job.active = True if request.form.get('active') == 'on' else False

    db.session.commit()
    return redirect(url_for('job_master.job_master'))


# ---------------------------------------------------------
# MOVE UP
# Admin / Management ONLY
# ---------------------------------------------------------
@job_master_bp.route('/job_master/move_up/<int:id>')
def job_master_move_up(id):

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    job = JobListMaster.query.get_or_404(id)
    above = JobListMaster.query.filter(JobListMaster.sort_order < job.sort_order)\
                               .order_by(JobListMaster.sort_order.desc()).first()

    if above:
        job.sort_order, above.sort_order = above.sort_order, job.sort_order
        db.session.commit()

    return redirect(url_for('job_master.job_master'))


# ---------------------------------------------------------
# MOVE DOWN
# Admin / Management ONLY
# ---------------------------------------------------------
@job_master_bp.route('/job_master/move_down/<int:id>')
def job_master_move_down(id):

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    job = JobListMaster.query.get_or_404(id)
    below = JobListMaster.query.filter(JobListMaster.sort_order > job.sort_order)\
                               .order_by(JobListMaster.sort_order.asc()).first()

    if below:
        job.sort_order, below.sort_order = below.sort_order, job.sort_order
        db.session.commit()

    return redirect(url_for('job_master.job_master'))
