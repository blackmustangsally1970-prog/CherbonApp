from flask import Blueprint, render_template

public_bp = Blueprint("public", __name__)

@public_bp.route('/viewing-pack')
def viewing_pack():

    return render_template(
        'weddings/viewing_pack.html'
    )