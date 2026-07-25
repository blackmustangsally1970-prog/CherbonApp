from app import db

class WeddingStaffAssignment(db.Model):
    __tablename__ = 'wedding_staff_assignment'

    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)

    role = db.Column(db.String(50))  # coordinator, wait, bar, floater, admin
    notes = db.Column(db.Text)
