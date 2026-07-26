from extensions import db

class JobListInstance(db.Model):
    __tablename__ = 'job_list_instance'

    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)

    # who owns it
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=True)

    master_id = db.Column(db.Integer, db.ForeignKey('job_list_master.id'))

    description = db.Column(db.Text)

    # NEW FIELDS
    in_progress_by = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=True)
    completed = db.Column(db.Boolean, default=False)
    completed_by = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=True)

    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
