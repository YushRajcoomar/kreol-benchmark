from transformers import MBart50Tokenizer
import numpy as np

def preprocess_function(examples,tokenizer,input_col='input',target_col='target'):
    inputs = examples[input_col]
    # outputs = [tgt_token + ' ' + x for x in examples['target']]
    outputs = examples[target_col]
    input_ids, attention_mask = tokenizer(inputs, max_length=128, truncation=True, padding="max_length").values()
    decoder_input_ids, decoder_attention_mask = tokenizer(text_target=outputs, max_length=128, truncation=True, padding="max_length").values()
    # input_tokenized = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    # output_tokenized = tokenizer(outputs, max_length=128, truncation=True, padding="max_length")
    # input_ids = input_tokenized["input_ids"]
    # attention_mask = input_tokenized["attention_mask"]   
    # decoder_input_ids = output_tokenized["input_ids"]
    # decoder_input_ids = np.array(output_tokenized["input_ids"].copy())
    # decoder_input_ids[:, 0] = tokenizer.bos_token_id
    # decoder_attention_mask = output_tokenized["attention_mask"]
    labels = decoder_input_ids.copy()
    # Set the labels to -100 for padding tokens
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask, "labels": labels}

