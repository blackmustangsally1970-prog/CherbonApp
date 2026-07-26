from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from extensions import db
from weddings.models import JobListMaster

job_master_bp = Blueprint('job_master', __name__, url_prefix='/weddings')

# ---------------------------------------------------------
# MASTER JOB LIST PAGE
# ---------------------------------------------------------
@job_master_bp.route('/job_master')
def job_master():
    jobs = JobListMaster.query.order_by(JobListMaster.sort_order.asc()).all()
    return render_template('weddings/job_master.html', jobs=jobs)


# ---------------------------------------------------------
# ADD MASTER JOB
# ---------------------------------------------------------
@job_master_bp.route('/job_master/add', methods=['POST'])
def job_master_add():
    role = request.form.get('role')
    description = request.form.get('description')

    if not description:
        flash("Description required", "danger")
        return redirect(url_for('job_master.job_master'))

    # new items go to bottom
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
# ---------------------------------------------------------
@job_master_bp.route('/job_master/delete/<int:job_id>')
def job_master_delete(job_id):
    job = JobListMaster.query.get_or_404(job_id)
    db.session.delete(job)
    db.session.commit()
    return redirect(url_for('job_master.job_master'))


# ---------------------------------------------------------
# UPDATE MASTER JOB
# ---------------------------------------------------------
@job_master_bp.route('/job_master/update/<int:job_id>', methods=['POST'])
def job_master_update(job_id):
    job = JobListMaster.query.get_or_404(job_id)

    job.role = request.form.get('role')
    job.description = request.form.get('description')
    job.active = True if request.form.get('active') == 'on' else False

    db.session.commit()
    return redirect(url_for('job_master.job_master'))


# ---------------------------------------------------------
# DRAG/DROP REORDER ENDPOINT
# ---------------------------------------------------------
@job_master_bp.route('/job_master/reorder', methods=['POST'])
def job_master_reorder():
    data = request.get_json()
    ids = data.get("order", [])

    for index, job_id in enumerate(ids):
        job = JobListMaster.query.get(int(job_id))
        job.sort_order = index + 1

    db.session.commit()
    return jsonify({"status": "ok"})
