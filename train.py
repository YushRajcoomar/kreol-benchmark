from pathlib import Path
from tokenizers.implementations import SentencePieceBPETokenizer
import pandas as pd
import numpy as np
import evaluate


from transformers import MBart50Tokenizer
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments #, Seq2SeqTrainer
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from transformers import logging

from data.data_utils.utils import preprocess_function

import os

TOKENIZER_VOCABULARY = 20000  # Total number of unique subwords the tokenizer can have

os.environ["WANDB_PROJECT"] = "Kreol - NMT"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 

def compute_metrics(eval_preds):
    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
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

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

tokenizer = MBart50Tokenizer.from_pretrained("/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok",max_len=256)

dataset = load_dataset(
    "json",
    data_files={'train':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_train.jsonl','test':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_test.jsonl',
                'val':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'}
)

tokenize_fn = preprocess_function(tokenizer)
dataset = dataset.map(tokenize_fn, batched=True)
train_dataset, test_dataset, val_dataset = dataset.values()

print("CUDA:", torch.cuda.is_available())

config = MBartConfig(vocab_size=TOKENIZER_VOCABULARY,max_position_embeddings=512,forced_eos_token_id=2)
model = MBartForConditionalGeneration(config)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir='./checkpoint_tests',
    num_train_epochs=4,
    per_gpu_train_batch_size=2,
    per_gpu_eval_batch_size=1,
    prediction_loss_only=True,
    report_to="wandb",
    run_name = "test_run_bleu",

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

trainer.train()