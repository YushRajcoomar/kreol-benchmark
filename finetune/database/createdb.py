from mkdb import db, Translation
import pandas as pd

new=False
# Create the database tables
if new:
    db.create_all()

src_lang = 'en'
tgt_lang = 'cr'

lang_pair_directory = f'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/{src_lang}-{tgt_lang}/{src_lang}-{tgt_lang}_train.jsonl'

data = pd.read_json(lang_pair_directory,lines=True)
# Create some example translations
translations_data = []

for idx,row in data.iterrows():
    translations_data.append(
        {
            'input_text':row['input'],
            'predicted_text':row['target'],
            'src_lang':src_lang,
            'tgt_lang':tgt_lang
        }
    )

# Insert the translations into the database
for data in translations_data:
    translation = Translation(input_text=data['input_text'], predicted_text=data['predicted_text'],src_lang=src_lang,tgt_lang=tgt_lang)
    db.session.add(translation)

# Commit the changes
db.session.commit()