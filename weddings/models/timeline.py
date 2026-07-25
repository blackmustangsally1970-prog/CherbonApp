from datetime import datetime
from extensions import db

class WeddingTimeline(db.Model):
    __tablename__ = 'wedding_timeline'

    id = db.Column(db.Integer, primary_key=True)

    wedding_id = db.Column(db.Integer, db.ForeignKey('wedding.id'), nullable=False)
    event_type = db.Column(db.String(50))  # email_sent, email_received, note, task_update, status_change
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
