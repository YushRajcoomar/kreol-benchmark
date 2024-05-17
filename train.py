import pandas as pd
from transformers import MBart50Tokenizer, AutoTokenizer
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from transformers import logging
import os
import logging
from data.data_utils.utils import preprocess_function, compute_metrics
import wandb
from transformers.integrations import WandbCallback
import numpy as np
from datasets import concatenate_datasets

LATEST_CHECKPOINTS = {
    "MBart50": "facebook/mbart-large-50-many-to-many-mmt",
    "MT5": "google/mt5-base",
    "NLLB200": "facebook/nllb-200-distilled-600M",
    "M2M100": "facebook/m2m100_418M",
    # "Aya101": "CohereForAI/aya-101" later
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.cuda.set_device(0)
os.environ["WANDB_PROJECT"] = "Kreol - NMT"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"  

### Parameters
TOKENIZER_PATH = "/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok" #from scratch
TOKENIZER_MAX_LEN = 128
TOKENIZER_VOCABULARY = 250055  # Total number of unique subwords the tokenizer can have
pretrained=True
bidirectional = False


### Training Parameters
num_epochs = 150
weight_decay = 0.1
fp16=True
param_config = {
    'epochs':num_epochs,
    'weight_decay':weight_decay,
    'fp16':fp16,
}

### Pretraining Model Parameters
model_name = "Aya101"
base_checkpoint = False
if base_checkpoint:
    checkpoint = LATEST_CHECKPOINTS[model_name]
    src_lang = "en"
    tgt_lang = "mfe"
else: #custom checkpoint
    checkpoint = "/home/yush/kreol-benchmark/checkpoint-120000_best" #maybe best model
    src_lang = "en_XX"
    tgt_lang = "cr_CR"

run_name = f"{model_name} ENxFR  en -> cr finetune_dict_sentences pick-high-save"

if pretrained:
    logging.info('Loading Pretrained MBART50 MMT EN-FR')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,src_lang=src_lang,tgt_lang=tgt_lang)#,max_len=TOKENIZER_MAX_LEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer_length = len(tokenizer) + 1 #tokenizer was modified to accomodate for new target lang
    model.resize_token_embeddings(tokenizer_length)
else:
    logging.info('Training from scratch')
    config = MBartConfig(vocab_size=TOKENIZER_VOCABULARY,max_position_embeddings=512,forced_eos_token_id=2,dropout=0.3)
    model = MBartForConditionalGeneration(config)
    tokenizer = MBart50Tokenizer.from_pretrained(TOKENIZER_PATH,max_len=TOKENIZER_MAX_LEN)

dataset = load_dataset(
    "json",
    data_files={
        'train':'/home/yush/kreol-benchmark/data/lang_data/en-cr/cr_en_sentences_train.json',
        'test':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_test.jsonl',
        'val':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'}
)

preprocessed_dataset = dataset.map(preprocess_function,fn_kwargs={'tokenizer':tokenizer}, batched=True)

train_dataset = preprocessed_dataset['train']
test_dataset = preprocessed_dataset['test']
val_dataset = preprocessed_dataset['val']


if bidirectional:
    tokenizer.src_lang = tgt_lang
    tokenizer.tgt_lang = src_lang
    bi_dataset = dataset.map(preprocess_function,fn_kwargs={'tokenizer':tokenizer,'input_col':'target','target_col':'input'}, batched=True)
    train_dataset = concatenate_datasets([train_dataset,bi_dataset['train']])
    test_dataset = concatenate_datasets([test_dataset,bi_dataset['test']])
    val_dataset = concatenate_datasets([val_dataset,bi_dataset['val']])

print("CUDA:", torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model
)

# Remove the 'report_to="wandb"' argument from Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./checkpoint/{model_name}',
    num_train_epochs=num_epochs,
    per_device_train_batch_size =48,
    per_device_eval_batch_size =4,
    include_inputs_for_metrics=True,
    prediction_loss_only=False,
    do_predict = True,
    weight_decay=weight_decay,
    evaluation_strategy='steps',
    eval_steps=10000,
    save_strategy = 'steps',
    save_steps=20000,
    load_best_model_at_end = False,
    metric_for_best_model= 'loss',
    greater_is_better = False,
    predict_with_generate = True,
    generation_num_beams = 4,
    generation_max_length = tokenizer_length,
    fp16=fp16,
    save_total_limit=10,
    label_smoothing_factor=0.1
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Initialize Wandb in 'dryrun' mode
wandb.init(name=run_name,config=param_config,mode='dryrun')

# Wrap the trainer with WandbCallback
trainer.add_callback(WandbCallback())

# Train the model
trainer.train()
