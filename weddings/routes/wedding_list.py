from flask import Blueprint, render_template
from app import db
from weddings.models import Wedding

wedding_list_bp = Blueprint('wedding_list', __name__, url_prefix='/weddings')

@wedding_list_bp.route('/')
def wedding_list():
    weddings = Wedding.query.order_by(Wedding.wedding_date.asc()).all()
    return render_template('weddings/wedding_list.html', weddings=weddings)
