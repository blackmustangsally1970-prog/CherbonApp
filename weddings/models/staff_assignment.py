from extensions import db
from models import Employee  # make sure this import exists

class WeddingStaffAssignment(db.Model):
    __tablename__ = 'wedding_staff_assignment'

    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)

    role = db.Column(db.String(50))  # coordinator, wait, bar, floater, admin
    notes = db.Column(db.Text)

    # ⭐ THIS LINE MAKES {{ s.employee.full_name }} WORK
    employee = db.relationship("Employee", backref="wedding_assignments")
