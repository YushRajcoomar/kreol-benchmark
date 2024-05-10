import pandas as pd

from sqlalchemy import inspect
from finetune import app,db
from finetune.models import Translation, User

app.app_context().push()

if not inspect(db.engine).has_table('user'):
    db.create_all()
    user = User(username='example_user', email='example@example.com', password='example_password')
    db.session.add(user)
    db.session.commit()
else:
    user = User.query.filter_by(email='example@example.com').first()

src_lang = 'en'
tgt_lang = 'cr'

lang_pair_directory = f'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/{src_lang}-{tgt_lang}/{src_lang}-{tgt_lang}_train.jsonl'

data = pd.read_json(lang_pair_directory,lines=True)
# Create some example translations
translations_data = []

for idx,row in data.iterrows():
    translations_data.append(
        {
            'index':row.name,
            'input_text':row['input'],
            'predicted_text':row['target'],
            'rating' : -1,
            'suggested_text': "",
            'src_lang':src_lang,
            'tgt_lang':tgt_lang,
            'user_id' : user.id
        }
    )

# Insert the translations into the database
for data in translations_data:
    translation = Translation(**data)
    db.session.add(translation)

# Commit the changes
db.session.commit()
print('db updated')