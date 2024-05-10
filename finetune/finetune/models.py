from finetune import db

class User(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.String(20),unique=True, nullable=False)
    email = db.Column(db.String(120),unique=True, nullable=False)
    password = db.Column(db.String(60),nullable=False)
    translations = db.relationship('Translation',backref='author',lazy=True)

    def __repr__(self):
        return f"User('{self.username}','{self.email}')"
    
class Translation(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    index = db.Column(db.String(255), nullable=False)
    input_text = db.Column(db.String(255), nullable=False)
    predicted_text = db.Column(db.String(255), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    suggested_text = db.Column(db.String(255), nullable=False)
    src_lang = db.Column(db.String(255), nullable=False)
    tgt_lang = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)


    def __repr__(self):
        return f"Translation('{self.input_text}','{self.predicted_text}','{self.rating}','{self.suggested_text},'{self.src_lang}','{self.tgt_lang}')"