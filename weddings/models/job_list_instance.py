from extensions import db


class JobListInstance(db.Model):
    __tablename__ = 'job_list_instance'

    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)

    # 🔥 NEW unified owner field
    account_id = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=True)

    master_id = db.Column(db.Integer, db.ForeignKey('job_list_master.id'))

    description = db.Column(db.Text)

    # 🔥 UPDATED unified fields
    in_progress_by = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=True)
    completed = db.Column(db.Boolean, default=False)
    completed_by = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=True)

    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    assigned_role = db.Column(db.String(50))
    status = db.Column(db.String(50), default="pending")

    # 🔥 NEW relationship to Account
    account = db.relationship("Account", foreign_keys=[account_id])
    in_progress_account = db.relationship("Account", foreign_keys=[in_progress_by])
    completed_account = db.relationship("Account", foreign_keys=[completed_by])
