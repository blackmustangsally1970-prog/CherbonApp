from flask import Blueprint, render_template, request, redirect, url_for, flash
from extensions import db
from auth.permissions import is_admin, is_coordinator, is_caterer
from weddings.models import Wedding, WeddingStaffAssignment
from models import Employee  # your existing staff table

staffing_bp = Blueprint('staffing', __name__, url_prefix='/weddings')


# ---------------------------------------------------------
# VIEW STAFF FOR A WEDDING
# Admin / Management / Coordinator (view-only)
# ---------------------------------------------------------
@staffing_bp.route('/<int:wedding_id>/staff')
def staff_list(wedding_id):

    # Caterers cannot access staffing
    if is_caterer():
        return "Unauthorized", 403

    wedding = Wedding.query.get_or_404(wedding_id)
    staff = WeddingStaffAssignment.query.filter_by(wedding_id=wedding.id).all()
    employees = Employee.query.order_by(Employee.full_name.asc()).all()

    return render_template(
        'weddings/staff.html',
        wedding=wedding,
        staff=staff,
        employees=employees,
        coordinator_view=is_coordinator()  # hide admin-only controls in template
    )


# ---------------------------------------------------------
# ADD STAFF TO WEDDING
# Admin / Management ONLY
# ---------------------------------------------------------
@staffing_bp.route('/<int:wedding_id>/staff/add', methods=['POST'])
def staff_add(wedding_id):

    # Coordinators cannot modify staff
    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    wedding = Wedding.query.get_or_404(wedding_id)

    employee_id = request.form.get('employee_id')
    role = request.form.get('role')

    if not employee_id:
        flash("Select a staff member", "danger")
        return redirect(url_for('staffing.staff_list', wedding_id=wedding_id))

    new_assignment = WeddingStaffAssignment(
        wedding_id=wedding.id,
        employee_id=employee_id,
        role=role
    )

    db.session.add(new_assignment)
    db.session.commit()

    return redirect(url_for('staffing.staff_list', wedding_id=wedding_id))


# ---------------------------------------------------------
# REMOVE STAFF FROM WEDDING
# Admin / Management ONLY
# ---------------------------------------------------------
@staffing_bp.route('/<int:wedding_id>/staff/remove/<int:assignment_id>')
def staff_remove(wedding_id, assignment_id):

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    assignment = WeddingStaffAssignment.query.get_or_404(assignment_id)
    db.session.delete(assignment)
    db.session.commit()

    return redirect(url_for('staffing.staff_list', wedding_id=wedding_id))


# ---------------------------------------------------------
# UPDATE STAFF ROLE
# Admin / Management ONLY
# ---------------------------------------------------------
@staffing_bp.route('/<int:wedding_id>/staff/update_role/<int:assignment_id>', methods=['POST'])
def staff_update_role(wedding_id, assignment_id):

    if is_coordinator() or is_caterer() or not is_admin():
        return "Unauthorized", 403

    assignment = WeddingStaffAssignment.query.get_or_404(assignment_id)
    new_role = request.form.get('role')

    assignment.role = new_role
    db.session.commit()

    return redirect(url_for('staffing.staff_list', wedding_id=wedding_id))
