from flask import Blueprint, render_template, redirect, url_for, request
from extensions import db
from permissions import admin_required, is_coordinator, is_caterer

from weddings.models import (
    Wedding,
    WeddingStaffAssignment,
    JobListInstance,
    JobListMaster,
    WeddingTimeline,
    WeddingDetailLibrary,
    Caterer
)

hub_bp = Blueprint('hub', __name__, url_prefix='/weddings')


# ---------------------------------------------------------
# WEDDING HUB (MAIN MENU FOR A SINGLE WEDDING)
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/hub')
def wedding_hub(wedding_id):

    # Caterers are NOT allowed to view coordinator/admin hub
    if is_caterer():
        return "Unauthorized", 403

    wedding = Wedding.query.get_or_404(wedding_id)

    # Coordinator CAN view hub, but cannot edit anything
    # Admin/Management have full access

    staff = WeddingStaffAssignment.query.filter_by(wedding_id=wedding.id).all()
    job_lists = JobListInstance.query.filter_by(wedding_id=wedding.id).all()
    timeline = WeddingTimeline.query.filter_by(wedding_id=wedding.id)\
                                    .order_by(WeddingTimeline.created_at.desc()).all()

    # detail templates (future client/caterer forms)
    detail_templates = WeddingDetailLibrary.query.filter_by(active=True)\
                                                 .order_by(WeddingDetailLibrary.default_order.asc()).all()

    caterers = Caterer.query.all()

    return render_template(
        'weddings/hub.html',
        wedding=wedding,
        staff=staff,
        job_lists=job_lists,
        timeline=timeline,
        detail_templates=detail_templates,
        caterers=caterers,
        coordinator_view=is_coordinator()  # allows template to hide admin-only controls
    )


# ---------------------------------------------------------
# QUICK LINK: GO TO HUB FROM CALENDAR
# ---------------------------------------------------------
@hub_bp.route('/hub_redirect/<int:wedding_id>')
def hub_redirect(wedding_id):
    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))


# ---------------------------------------------------------
# SET CATERER (ADMIN/MANAGEMENT ONLY)
# ---------------------------------------------------------
@hub_bp.route('/<int:wedding_id>/set_caterer', methods=['POST'])
@admin_required
def set_caterer(wedding_id):

    wedding = Wedding.query.get_or_404(wedding_id)
    wedding.caterer_id = request.form['caterer_id']
    db.session.commit()

    return redirect(url_for('hub.wedding_hub', wedding_id=wedding_id))
