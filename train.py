from pathlib import Path
from tokenizers.implementations import SentencePieceBPETokenizer
import pandas as pd
from transformers import MBart50Tokenizer, AutoTokenizer
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments #, Seq2SeqTrainer
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from transformers import logging
import os
import logging
from data.data_utils.utils import preprocess_function
import wandb
from transformers.integrations import WandbCallback
from datasets import concatenate_datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logging.set_verbosity_warning()
# logging.enable_progress_bar()


# from transformers import TrainingCallback

# class BestModelCheckpointCallback(TrainingCallback):
#     def __init__(self, metric_to_track, greater_is_better=True):
#         self.metric_to_track = metric_to_track
#         self.greater_is_better = greater_is_better
#         self.best_score = None
#         self.best_checkpoint_dir = None

#     def on_save(self, args, state, control):
#         # Access metrics from the current evaluation step
#         metrics = state.eval_metrics

#         if self.metric_to_track not in metrics:
#             raise ValueError(f"Metric '{self.metric_to_track}' not found in evaluation metrics.")

#         current_score = metrics[self.metric_to_track]

#         if self.best_score is None or (
#             (self.greater_is_better and current_score > self.best_score) or
#             (not self.greater_is_better and current_score < self.best_score)
#         ):
#             self.best_score = current_score
#             self.best_checkpoint_dir = args.output_dir  # Update checkpoint directory

#             # Upload the best checkpoint to WandB
#             model = Trainer.get_saved_model(args.output_dir)
#             wandb.log({"model": model})  # or "checkpoint" depending on your setup

#             print(f"Saved new best checkpoint with {self.metric_to_track}: {current_score}")



import evaluate
import numpy as np
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

os.environ["WANDB_PROJECT"] = "Kreol - NMT"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"  

### Parameters
# TOKENIZER_PATH = "/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok"
TOKENIZER_MAX_LEN = 128
TOKENIZER_VOCABULARY = 250055  # Total number of unique subwords the tokenizer can have
pretrained=True
bidirectional = True
src_lang = "en_XX"
tgt_lang = "cr_CR"

### Training Parameters
num_epochs = 100
weight_decay = 0.1
fp16=True
param_config = {
    'epochs':num_epochs,
    'weight_decay':weight_decay,
    'fp16':fp16,
}
run_name = "MbartPT ENxFR  en <--> cr epoch100 weightdecay0.1 labelsmoothing0.1 [good]"

config = MBartConfig(vocab_size=TOKENIZER_VOCABULARY,max_position_embeddings=512,forced_eos_token_id=2,dropout=0.3)
if pretrained:
    logging.info('Loading Pretrained MBART50 MMT EN-FR')
    checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,src_lang=src_lang,tgt_lang=tgt_lang,max_len=TOKENIZER_MAX_LEN)
    model = MBartForConditionalGeneration.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
else:
    logging.info('Training from scratch')
    model = MBartForConditionalGeneration(config)
    tokenizer = MBart50Tokenizer.from_pretrained(TOKENIZER_PATH,max_len=TOKENIZER_MAX_LEN)

# tgt_lang = "<cr_CR>"
# tokenizer.add_tokens(tgt_lang,special_tokens=True)



dataset = load_dataset(
    "json",
    data_files={'train':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_train.jsonl','test':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_test.jsonl',
                'val':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'}
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

from transformers.trainer_utils import IntervalStrategy

# Remove the 'report_to="wandb"' argument from Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./checkpoint',
    num_train_epochs=num_epochs,
    per_device_train_batch_size =16,
    per_device_eval_batch_size =4,
    include_inputs_for_metrics=True,
    prediction_loss_only=False,
    do_predict = True,
    weight_decay=weight_decay,
    evaluation_strategy='steps',
    eval_steps=10000,
    save_strategy = 'steps',
    save_steps=30000,
    load_best_model_at_end = False,
    metric_for_best_model= 'loss',
    greater_is_better = False,
    predict_with_generate = True,
    generation_num_beams = 4,
    generation_max_length = TOKENIZER_MAX_LEN,
    fp16=fp16,
    save_total_limit=3,
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

# Wrap the trainer with WandbCallback
trainer.add_callback(WandbCallback())

# Train the model
trainer.train()
