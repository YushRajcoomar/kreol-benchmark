import pandas as pd
from transformers import MBart50Tokenizer, AutoTokenizer
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from transformers import logging
import os
import logging
from data.data_utils.utils import preprocess_function  #, compute_metrics
import wandb
from transformers.integrations import WandbCallback
import numpy as np
from datasets import concatenate_datasets
from data.data_utils.prepare_data import DataPreparation 

import evaluate


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

LATEST_CHECKPOINTS = {
    "MBart50": "facebook/mbart-large-50-many-to-many-mmt",
    "MT5": "google/mt5-base",
    "NLLB200": "facebook/nllb-200-distilled-600M",
    "M2M100": "facebook/m2m100_418M",
    # "Aya101": "CohereForAI/aya-101" later
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["WANDB_PROJECT"] = "Kreol - NMT"  # name your W&B project

### Parameters
TOKENIZER_PATH = "/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok" #from scratch
TOKENIZER_MAX_LEN = 128
TOKENIZER_VOCABULARY = 250055  # Total number of unique subwords the tokenizer can have
pretrained=True
bidirectional = False


### Training Parameters
num_epochs = 3
weight_decay = 0.1
fp16=True
param_config = {
    'epochs':num_epochs,
    'weight_decay':weight_decay,
    'fp16':fp16,
}

### Pretraining Model Parameters
model_name = "MBart50"
base_checkpoint = False
if base_checkpoint:
    checkpoint = LATEST_CHECKPOINTS[model_name]
    src_lang = "en"
    tgt_lang = "mfe"
else: #custom checkpoint
    checkpoint = "/home/yush/kreol-benchmark/checkpoint_tests/checkpoint-11_best500ft" #maybe best model
    # checkpoint = 'facebook/mbart-large-50-many-to-many-mmt'
    src_lang = "en_XX"
    tgt_lang = "cr_CR"

run_name = f"{model_name} en <-> cr sentences_hq500 finetune1"

if pretrained:
    logging.info(f'Loading Pretrained {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,src_lang=src_lang,tgt_lang=tgt_lang)#,max_len=TOKENIZER_MAX_LEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # tokenizer_length = len(tokenizer)
    model.resize_token_embeddings(TOKENIZER_VOCABULARY)
else:
    logging.info('Training from scratch')
    config = MBartConfig(vocab_size=TOKENIZER_VOCABULARY,max_position_embeddings=512,forced_eos_token_id=2,dropout=0.3)
    model = MBartForConditionalGeneration(config)
    tokenizer = MBart50Tokenizer.from_pretrained(TOKENIZER_PATH,max_len=TOKENIZER_MAX_LEN)


data_preparation = DataPreparation(
    dataset_paths={
        'train':'/home/yush/kreol-benchmark/data/lang_data/en-cr/finetune300_vlong_flores.jsonl',
        'test':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_test.jsonl',
        'val':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'},
        # 'val':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'},
    tokenizer=tokenizer,
    bidirectional=False,
    src_lang="en_XX",
    tgt_lang="cr_CR",
    rating_adapter=True
)

mixer_dict = {
    'hq500':['/home/yush/kreol-benchmark/data/lang_data/en-cr/hq_sentences/cr_en_hq_gptapi_500.json', 0.1]}
#     # 'dict':['/home/yush/kreol-benchmark/data_collection/notebooks/cr_en_dict_sentences.json', 0.3],
#     'train_base':['/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_train.jsonl', 0.1],
# }

data_preparation.load_data()
data_preparation.preprocess_data()
data_preparation.mix_data(mixer_dict,train_only=True)

train_dataset = data_preparation.train_dataset
test_dataset = data_preparation.test_dataset


print("CUDA:", torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model
)

# Remove the 'report_to="wandb"' argument from Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./checkpoint/{model_name}/bidirectional/finetune/hq500_punc_1',
    num_train_epochs=num_epochs,
    per_device_train_batch_size =48,
    per_device_eval_batch_size =4,
    include_inputs_for_metrics=True,
    prediction_loss_only=False,
    do_predict = True,
    weight_decay=weight_decay,
    evaluation_strategy='steps',
    eval_steps=10,
    save_strategy = 'steps',
    save_steps=20000,
    load_best_model_at_end = False,
    metric_for_best_model= 'loss',
    greater_is_better = False,
    predict_with_generate = True,
    generation_num_beams = 4,
    generation_max_length = TOKENIZER_VOCABULARY,
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
wandb.init(name=run_name,config=param_config)

# Train the model
trainer.train()
