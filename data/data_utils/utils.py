from transformers import MBart50Tokenizer
import numpy as np
import evaluate

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


def compute_metrics(eval_preds):
    metrics = {}
    # Load SacreBLEU
    sacrebleu_metric = evaluate.load("sacrebleu")

    # Load ROUGE
    rouge_metric = evaluate.load("rouge")

    # Load chrF
    chrf_metric = evaluate.load("chrf")
    preds, labels, inputs = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    sacrebleu_result = sacrebleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

    ### rename dicts
    sacrebleu_result['sacrebleu_score'] = sacrebleu_result.pop('score')
    ngram_precisions = sacrebleu_result.pop('precisions')
    sacrebleu_result['precision_1_gram'] = ngram_precisions[0]
    sacrebleu_result['precision_2_gram'] = ngram_precisions[1]
    sacrebleu_result['precision_3_gram'] = ngram_precisions[2]
    sacrebleu_result['precision_4_gram'] = ngram_precisions[3]
    chrf_result['chrf_score'] = chrf_result.pop('score')

    # {"bleu": result["score"]}
    metrics.update(sacrebleu_result)
    metrics.update(rouge_result)
    metrics.update(chrf_result)
    return metrics