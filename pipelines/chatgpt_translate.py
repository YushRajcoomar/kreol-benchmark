import pandas as pd
from tqdm import tqdm
import requests
import pickle


### DATA
cr_mono = pd.read_json('/home/yush/kreol-benchmark/data/lang_data/cr/cr_train.jsonl',lines=True)
cr_mono = [x for x in cr_mono['input'] if len(x.split(' ')) > 8]

### SAVE PATH 
output_file_path = '/home/yush/kreol-benchmark/data_collection/data_extracted/raw/cr_en_dabretrain_parallel_30k.pkl'

#API STUFF
api_key = "<API KEY>"
endpoint = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
warm_up_msgs = [
        {"role": "user", "content": "You will act as my translator. I speak English but i was given a script in Mauritian Creole. "},
        {"role": "assistant", "content": "Sure! Please provide the lines of text you'd like me to translate, one line at a time."},
        {"role": "user", "content":"input: BILAN BANN LAPROS EK LETID LOR GRAMER"},
        {"role": "assistant", "content": "SUMMARY OF EXERCISES AND STUDIES ON GRAMMAR"},
        {"role": "user", "content":"input:  ANT LAPROS TRADISIONEL EK LAPROS MODERN"},
        {"role": "assistant", "content":"BETWEEN TRADITIONAL APPROACHES AND MODERN APPROACHES"},]


data = {
    "model": "gpt-3.5-turbo-0125",
    "messages": warm_up_msgs,
    "max_tokens": 4096
}

total_completions = ''
parallel_string = []
for idx in tqdm(range(0,33730,10),desc="Batches"):
    translate_string = ''
    msgs_to_translate = cr_mono[idx:idx+10]
    for msg in msgs_to_translate:
        translate_string += f'<sep> {msg} <sep>'

    warm_up_msgs = [
        {"role": "user", "content": "You will act as my translator. I speak English but i was given a script in Mauritian Creole. "},
        {"role": "assistant", "content": "Sure! Please provide the lines of text you'd like me to translate, one line at a time."},
        {"role": "user", "content":"In the next prompt, I will pass Mauritian creole data with a separator tag <sep>. Please keep the tag in the output text in the correct place. This is of the utmost importance"},
        {"role": "assistant", "content":":Will do, please provide the sentences in Mauritian Creole, and I will translate in English"},
        {"role": "user", "content":"<sep> Mo p gagne extra faim. <sep> Mo bizin retourne lakaz. <sep> Mo mank mo ban parents et mo ban zanimo. <sep>"},
        {"role":"assistant","content":'<sep> I am extremely hungry. <sep> i need to get back home. <sep> I miss my parents and my pets. <sep>'},
    ]

    warm_up_msgs.append({"role":"user","content":translate_string})

    data = {
        "model": "gpt-3.5-turbo-0125",
        "messages": warm_up_msgs,
        "max_tokens": 4096
    }

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        completion = response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Failed with status code {response.status_code}: {response.text}")

    # total_completions += f'<batch_sep> {completion} <batch_sep>'
    parallel_string.append((translate_string,completion))
    if not((idx+10) % 100):
        
        with open(output_file_path, 'wb') as file:
            pickle.dump(parallel_string, file)
        print(f"{idx+10} sentences saved")
