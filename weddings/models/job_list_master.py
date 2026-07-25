from app import db

class JobListMaster(db.Model):
    __tablename__ = 'job_list_master'

    id = db.Column(db.Integer, primary_key=True)

    role = db.Column(db.String(50))  # coordinator, wait, bar
    description = db.Column(db.Text, nullable=False)
    sort_order = db.Column(db.Integer, default=0)
    active = db.Column(db.Boolean, default=True)