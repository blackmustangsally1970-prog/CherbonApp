from app import db

class WeddingDetailLibrary(db.Model):
    __tablename__ = 'wedding_detail_library'

    id = db.Column(db.Integer, primary_key=True)

    category = db.Column(db.String(50), nullable=False)   # ceremony, reception, menu, setup, timeline, admin
    field_name = db.Column(db.Text, nullable=False)
    field_type = db.Column(db.String(20), nullable=False) # text, textarea, number, dropdown, checkbox

    default_value = db.Column(db.Text)
    coordinator_visible = db.Column(db.Boolean, default=True)
    staff_visible = db.Column(db.Boolean, default=True)

    default_order = db.Column(db.Integer, default=0)
    default_notes = db.Column(db.Text)
    active = db.Column(db.Boolean, default=True)

    updated_at = db.Column(db.DateTime, server_default=db.func.now())
