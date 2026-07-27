from flask import Blueprint, render_template
from extensions import db
from permissions import is_admin, is_coordinator, is_caterer
from weddings.models import Wedding

wedding_list_bp = Blueprint('wedding_list', __name__, url_prefix='/weddings')


@wedding_list_bp.route('/')
def wedding_list():

    # Coordinator cannot access wedding list
    if is_coordinator():
        return "Unauthorized", 403

    # Caterer cannot access wedding list
    if is_caterer():
        return "Unauthorized", 403

    # Admin + Management only
    if not is_admin():
        return "Unauthorized", 403

    weddings = Wedding.query.order_by(Wedding.wedding_date.asc()).all()

    return render_template('weddings/wedding_list.html', weddings=weddings)
