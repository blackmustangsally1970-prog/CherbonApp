from datetime import datetime
from extensions import db

class InspirationCategory(db.Model):
    __tablename__ = 'inspiration_categories'

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    description = db.Column(db.Text)

    display_order = db.Column(db.Integer, default=0)

    active = db.Column(db.Boolean, default=True)

    items = db.relationship(
        "InspirationItem",
        backref="category",
        lazy=True
    )


class InspirationItem(db.Model):
    __tablename__ = 'inspiration_items'

    id = db.Column(db.Integer, primary_key=True)

    category_id = db.Column(
        db.Integer,
        db.ForeignKey('inspiration_categories.id'),
        nullable=False
    )

    title = db.Column(db.String(200), nullable=False)

    slug = db.Column(
        db.String(255),
        unique=True,
        index=True
    )

    short_description = db.Column(db.Text)

    price_from = db.Column(db.Numeric(10, 2))

    featured = db.Column(db.Boolean, default=False)

    active = db.Column(db.Boolean, default=True)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    photos = db.relationship(
        "InspirationPhoto",
        backref="item",
        lazy=True,
        cascade="all, delete-orphan"
    )


class InspirationPhoto(db.Model):
    __tablename__ = 'inspiration_photos'

    id = db.Column(db.Integer, primary_key=True)

    inspiration_id = db.Column(
        db.Integer,
        db.ForeignKey('inspiration_items.id'),
        nullable=False
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    caption = db.Column(db.String(255))

    display_order = db.Column(
        db.Integer,
        default=0
    )

    featured = db.Column(
        db.Boolean,
        default=False
    )

    uploaded_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )
