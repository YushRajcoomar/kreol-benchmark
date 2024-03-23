from transformers import MBart50Tokenizer

def preprocess_function(examples,tokenizer_path,tokenizer_max_length):
    tokenizer = MBart50Tokenizer.from_pretrained(tokenizer_path,max_len=tokenizer_max_length)
    inputs = examples['input']
    outputs = examples['target']
    input_tokenized = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    output_tokenized = tokenizer(outputs, max_length=128, truncation=True, padding="max_length")
    input_ids = input_tokenized["input_ids"]
    attention_mask = input_tokenized["attention_mask"]
    decoder_input_ids = output_tokenized["input_ids"]
    decoder_attention_mask = output_tokenized["attention_mask"]
    labels = decoder_input_ids.copy()
    # Set the labels to -100 for padding tokens
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask, "labels": labels}

