from flask import Blueprint, render_template, request
from extensions import db
from weddings.models.appointments import Appointment

public_bp = Blueprint("public", __name__)

@public_bp.route("/appointments", methods=["GET", "POST"])
def appointments_public():
    # your existing logic here
    ...
