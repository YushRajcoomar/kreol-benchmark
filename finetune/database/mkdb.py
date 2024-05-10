from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///translations.db'
db = SQLAlchemy(app)

# Define your SQLAlchemy model
class Translation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(255))
    predicted_text = db.Column(db.String(255))
    rating = db.Column(db.Integer)
    suggested_text = db.Column(db.String(255))
    src_lang = db.Column(db.String(255))
    tgt_lang = db.Column(db.String(255))