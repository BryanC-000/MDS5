from db import db

class Img(db.Model):
    # id = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.LargeBinary, nullable=False)
    name = db.Column(db.String, nullable=False, primary_key=True)
    mimetype = db.Column(db.String, nullable=False)