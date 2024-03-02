from pathlib import Path
from tokenizers.implementations import SentencePieceBPETokenizer
import pandas as pd
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

tokenizer = MBart50Tokenizer.from_pretrained("/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok",max_len=256)

dataset = load_dataset(
    "json",
    data_files={'train':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cren-cr_train.jsonl','test':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cren-cr_test.jsonl',
                'val':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cren-cr_dev.jsonl'}
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
    output_dir='./checkpoint',
    num_train_epochs=200,
    per_gpu_train_batch_size=32,
    per_gpu_eval_batch_size=4,
    prediction_loss_only=True,
    report_to="wandb",
    run_name = "<insert name>"
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()