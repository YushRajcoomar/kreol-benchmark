from pathlib import Path
from tokenizers.implementations import SentencePieceBPETokenizer
import pandas as pd
from transformers import MBart50Tokenizer
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments #, Seq2SeqTrainer
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from transformers import logging

# logging.set_verbosity_warning()
# logging.enable_progress_bar()

def preprocess_function(examples):
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


TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
MAX_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

MODEL_CHECKPOINT = "mbart-large-50"  # Name of pretrained model from 🤗 Model Hub


tokenizer = MBart50Tokenizer.from_pretrained("/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok",max_len=256)

dataset = load_dataset(
    "json",
    data_files={'train':'/mnt/disk/yrajcoomar/kreol-benchmark/experiments/data/en-cr/en-cr_train.jsonl','test':'/mnt/disk/yrajcoomar/kreol-benchmark/experiments/data/en-cr/en-cr_test.jsonl',
                'val':'/mnt/disk/yrajcoomar/kreol-benchmark/experiments/data/en-cr/en-cr_dev.jsonl'}
)
dataset = dataset.map(preprocess_function, batched=True)

train_dataset = dataset['train']
test_dataset = dataset['test']
val_dataset = dataset['val']

print("CUDA:", torch.cuda.is_available())

config = MBartConfig(vocab_size=TOKENIZER_VOCABULARY,max_position_embeddings=512,forced_eos_token_id=2)
model = MBartForConditionalGeneration(config)


collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir='./checkpoint',
    num_train_epochs=100,
    per_gpu_train_batch_size=32,
    per_gpu_eval_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()