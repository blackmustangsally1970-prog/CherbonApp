from extensions import db
from cherbonapp.models import Account  # NEW unified account model

class WeddingStaffAssignment(db.Model):
    __tablename__ = 'wedding_staff_assignment'

    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)

    # 🔥 NEW FIELD: unified account foreign key
    account_id = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=False)

    role = db.Column(db.String(50))  # coordinator, wait, bar, floater, admin
    notes = db.Column(db.Text)

    # 🔥 NEW RELATIONSHIP: link to Account instead of Employee
    account = db.relationship("Account", backref="wedding_assignments")
