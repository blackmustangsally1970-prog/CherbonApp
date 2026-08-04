from extensions import db

class LegacyAccountMap(db.Model):
    __tablename__ = "legacy_account_map"

    id = db.Column(db.Integer, primary_key=True)

    legacy_table = db.Column(db.String(50), nullable=False)
    legacy_id = db.Column(db.Integer, nullable=False)

    account_id = db.Column(db.Integer, db.ForeignKey("accounts.id"), nullable=False)
