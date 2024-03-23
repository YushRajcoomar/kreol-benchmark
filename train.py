from pathlib import Path
from tokenizers.implementations import SentencePieceBPETokenizer
import pandas as pd
from transformers import MBart50Tokenizer
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments #, Seq2SeqTrainer
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from transformers import logging
import os
from data.data_utils.utils import preprocess_function


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


os.environ["WANDB_PROJECT"] = "Kreol - NMT"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
TOKENIZER_PATH = "/mnt/disk/yrajcoomar/kreol-benchmark/pipelines/tok"
TOKENIZER_MAX_LEN = 128
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have


tokenizer = MBart50Tokenizer.from_pretrained(TOKENIZER_PATH,max_len=TOKENIZER_MAX_LEN)

dataset = load_dataset(
    "json",
    data_files={'train':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_train.jsonl','test':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_test.jsonl',
                'val':'/mnt/disk/yrajcoomar/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'}
)

dataset = dataset.map(preprocess_function,fn_kwargs={'tokenizer_path':TOKENIZER_PATH,'tokenizer_max_length':TOKENIZER_MAX_LEN}, batched=True)

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
    per_device_train_batch_size =32,
    per_device_eval_batch_size =4,
    prediction_loss_only=True,
    report_to="wandb",
    run_name = "fp_16-x-weight_decay-0.1",
    do_predict = True,
    weight_decay=0.1,
    evaluation_strategy='steps',
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end = True,
    metric_for_best_model= 'loss',
    greater_is_better = False,
    predict_with_generate = True,
    generation_num_beams = 4, 
    generation_max_length = TOKENIZER_MAX_LEN,
    fp16=True,
    save_total_limit=1,
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

trainer.train()
