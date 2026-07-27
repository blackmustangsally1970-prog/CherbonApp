from flask import Blueprint, render_template, session
from datetime import date
from extensions import db
from permissions import is_admin, is_caterer
from weddings.models import Wedding, WeddingTimeline

caterer_bp = Blueprint('caterer', __name__, url_prefix='/caterer')


# ---------------------------------------------------------
# CATERER CALENDAR (FILTERED BY caterer_id)
# ---------------------------------------------------------
@caterer_bp.route('/calendar')
def caterer_calendar():

    # Caterer OR Admin/Management can view
    if not (is_caterer() or is_admin()):
        return "Unauthorized", 403

    caterer_id = session.get('caterer_id')
    today = date.today()

    weddings = Wedding.query.filter(
        Wedding.date >= today,
        Wedding.caterer_id == caterer_id
    ).order_by(Wedding.date.asc()).all()

    return render_template(
        'weddings/caterer/calendar.html',
        weddings=weddings,
        caterer_name=session.get('caterer_name')
    )


# ---------------------------------------------------------
# SAFETY CHECK FOR ANY CATERER PAGE
# ---------------------------------------------------------
def caterer_guard(wedding):
    # Admin/Management bypass guard
    if is_admin():
        return True

    # Caterer must match assigned wedding
    if is_caterer() and wedding.caterer_id == session.get('caterer_id'):
        return True

    return False


# ---------------------------------------------------------
# CATERER MENU (READ ONLY)
# ---------------------------------------------------------
@caterer_bp.route('/menu/<int:wedding_id>')
def caterer_menu(wedding_id):
    wedding = Wedding.query.get_or_404(wedding_id)

    if not caterer_guard(wedding):
        return "Unauthorized", 403

    menu_items = wedding.menu_items
    dietary = wedding.dietary_requirements

    return render_template(
        'weddings/caterer/menu.html',
        wedding=wedding,
        menu_items=menu_items,
        dietary=dietary
    )


# ---------------------------------------------------------
# CATERER TIMELINE (READ ONLY)
# ---------------------------------------------------------
@caterer_bp.route('/timeline/<int:wedding_id>')
def caterer_timeline(wedding_id):
    wedding = Wedding.query.get_or_404(wedding_id)

    if not caterer_guard(wedding):
        return "Unauthorized", 403

    timeline = WeddingTimeline.query.filter_by(wedding_id=wedding.id)\
                                    .order_by(WeddingTimeline.created_at.asc()).all()

    return render_template(
        'weddings/caterer/timeline.html',
        wedding=wedding,
        timeline=timeline
    )
