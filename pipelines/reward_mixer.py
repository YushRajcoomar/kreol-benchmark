# %%
import pandas as pd

# %%
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'/home/yush/kreol-benchmark/')

# %%

from transformers import MBartForConditionalGeneration, MBart50Tokenizer, MBartConfig
import torch.nn as nn
import torch
import os

# import json

# data = [{"input_text": "Hello, how are you?", "predicted_text": "Bonzour, kouma to ete?", "perfect_text": "Bonzour, kouma sava?", "rating": "4"},
# {"input_text": "What is your name?", "predicted_text": "Kouma to apel?", "perfect_text": "Kouma to apele?", "rating": "4"},
# {"input_text": "I am going to the market.", "predicted_text": "Mo pe al bazar.", "perfect_text": "Mo pe al bazar.", "rating": "5"},
# {"input_text": "Please sit down.", "predicted_text": "Si sa a plai ou, asize.", "perfect_text": "S'il vous plait, asize.", "rating": "2"},
# {"input_text": "Thank you very much.", "predicted_text": "Mersi boukou.", "perfect_text": "Mersi boukou.", "rating": "5"}]

# with open('/home/yush/kreol-benchmark/pipelines/reward_model_data/dummy.jsonl', 'w') as f:
#     for item in data:
#         # Write dictionary to file on new line'
#         item_dict = {'input': item['input_text'], 'target': item['perfect_text'], 'predicted_text':item['predicted_text'],'rating':item['rating']}
#         f.write(json.dumps(item_dict) + '\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomMBartForConditionalGeneration(MBartForConditionalGeneration):
    def __init__(self, config: MBartConfig):
        super().__init__(config)
        # Additional layer for processing the predicted text, perfect text, and rating
        self.additional_layer = nn.Linear(config.d_model, config.d_model)
        self.rating_layer = nn.Linear(1, config.d_model)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, 
                predicted_text_ids=None, rating=None):
        # Compute the original mBART outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels #perfect completion
        )

        # Process additional inputs
        if predicted_text_ids is not None and rating is not None:
            predicted_embeds = self.model.encoder(predicted_text_ids)[0]
            perfect_embeds = self.model.encoder(labels)[0] # [n,12,1024]
            rating_embeds = self.rating_layer(rating.unsqueeze(1).float()).unsqueeze(1) # [n,1,1024]

            # not sure what's up
            combined_embeds = predicted_embeds + perfect_embeds + rating_embeds
            combined_embeds = self.additional_layer(combined_embeds)

            # Calculate custom loss
            sigma = 1
            custom_loss = self.custom_loss_function(predicted_embeds, perfect_embeds, rating.unsqueeze(1).float(), outputs.loss.item(), sigma)
            # print(f"Output Loss: {outputs.loss}, Custom Loss : {custom_loss}")
            outputs.loss += custom_loss

        return outputs
    
    def custom_loss_function_simple(self,predicted_embeds, perfect_embeds, rating, loss, sigma):
        """
        In this loss function we simply divide the loss by the rating, scale it with some regularizer term sigma, and add it back to the model.
        The idea is that the model is penalized harsher for its loss on a model with a small rating. 
        We assume that poorly rated pairs are likely to be poorly translated by our own model, thus need harsher penalty.
        """
        loss_added = torch.divide(loss,rating[:,None,:]).mean() * sigma
        return loss_added
    
    def custom_loss_function_error_embedding(self,predicted_embeds, perfect_embeds, rating, loss, sigma):
        """
        In this loss function we use the mean absolute difference between the encoder embedding of the predicted embedding and the perfect embedding.
        We divide said average l1 loss with the rating, thus poorly rated translations are harshly penalized by some scale of the l1 error.
        We maintain the sigma regularizer to prevent overfitting.
        """
        l1_loss = abs(predicted_embeds - perfect_embeds)
        scaled_l1_loss = torch.divide(l1_loss,rating[:,None,:]).mean() * sigma
        return scaled_l1_loss
    
    def custom_loss_function_combined(self,predicted_embeds, perfect_embeds, rating, loss, sigma):
        """
        Combination of simple and error embedding
        """
        l1_loss = abs(predicted_embeds - perfect_embeds) #[bs,128,1024]
        scaled_l1_loss = torch.divide(l1_loss,rating[:,None,:]).mean() #[bs,128,1024] / [[bs,1,1]]
        loss_added = torch.divide(loss,rating[:,None,:]).mean()
        total_loss = (scaled_l1_loss + loss_added) * sigma
        return total_loss


    def custom_loss_function(self, predicted_embeds, perfect_embeds, rating,loss, sigma):
        # Define your custom loss calculation
        sq_loss = ((predicted_embeds - perfect_embeds) ** 2)
        scaled_sq_loss = torch.divide(sq_loss,rating[:,None,:])
        mse_loss = scaled_sq_loss.mean() * sigma
        return mse_loss


ckpt = '/home/yush/kreol-benchmark/checkpoint_tests/checkpoint-11_best500ft'
tokenizer = MBart50Tokenizer.from_pretrained(ckpt)
model = CustomMBartForConditionalGeneration.from_pretrained(ckpt)
custom_model = CustomMBartForConditionalGeneration.from_pretrained(ckpt).to(device)
custom_model.load_state_dict(model.state_dict(), strict=False)



# %%
from data.data_utils.prepare_data import DataPreparation 

data_preparation = DataPreparation(
    dataset_paths={
        'train':'/home/yush/kreol-benchmark/rated_translations.jsonl'},
        # 'val':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'},
    tokenizer=tokenizer,
    bidirectional=False,
    src_lang="en_XX",
    tgt_lang="cr_CR",
    rating_adapter=True
)

mixer_dict = {
    'hq500':['/home/yush/kreol-benchmark/data/lang_data/en-cr/hq_sentences/cr_en_hq_gptapi_500.json', 0.2],
    # 'dict':['/home/yush/kreol-benchmark/data_collection/notebooks/cr_en_dict_sentences.json', 0.3],
    'train_base':['/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_train.jsonl', 0.1],
}

data_preparation.load_data()
data_preparation.preprocess_data()
data_preparation.mix_data(mixer_dict,train_only=True)

train_dataset = data_preparation.train_dataset

# %%

# from datasets import load_dataset


# # Load dataset from JSONL file
# dataset = load_dataset('json', data_files='/home/yush/kreol-benchmark/rated_translations.jsonl')

# def preprocess_function(examples):
#     return {
#         'input': examples['input'],
#         'rating': examples['rating'],
#         'predicted_text': examples['predicted_text'],
#         'target': examples['suggested_text'], #perfect
        
#     }

# tokenized_dataset = dataset.map(preprocess_function, batched=True)

# %%

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
BATCH_SIZE=16
NUM_EPOCHS=1
TOTAL_STEPS = len(train_dataset) // BATCH_SIZE * NUM_EPOCHS

training_args = Seq2SeqTrainingArguments(
    output_dir='/home/yush/kreol-benchmark/pipelines/reward_model_data/results',
    # evaluation_strategy='epoch',
    per_device_train_batch_size=BATCH_SIZE,
    # per_device_eval_batch_size =4,
    # per_device_eval_batch_size=16,
    save_total_limit=3,
    save_steps = TOTAL_STEPS,
    num_train_epochs=NUM_EPOCHS,
    # predict_with_generate=True,
    remove_unused_columns=False,
    # do_eval=True,
    # evaluation_strategy='epoch',
    logging_dir='/home/yush/kreol-benchmark/pipelines/reward_model_data/logs',
    # label_smoothing_factor=0.1
)



# Define a data collator that handles the additional inputs
class DataCollatorWithAdditionalInputs:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [example['input'] for example in batch]
        predicted_texts = [example['predicted_text'] for example in batch]
        perfect_texts = [example['target'] for example in batch]
        ratings = [int(example['rating']) for example in batch]

        # Tokenize texts
        input_encodings, input_attention_mask = self.tokenizer(input_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt").values()
        predicted_encodings, predicted_enc_attention_mask = self.tokenizer(predicted_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt").values()
        decoder_input_ids, decoder_attention_mask = self.tokenizer(perfect_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt").values()

        # Convert ratings to tensor
        ratings_tensor = torch.tensor(ratings, dtype=torch.int64)

        batch_encodings = {
            'input_ids': input_encodings,
            'attention_mask': input_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            # 'decoder_attention_mask': decoder_attention_mask,
            'labels': decoder_input_ids,
            'predicted_text_ids': predicted_encodings,
            # 'predicted_attention_mask':predicted_enc_attention_mask,

            'rating': ratings_tensor
        }
        return batch_encodings

data_collator = DataCollatorWithAdditionalInputs(tokenizer)

os.environ['WANDB_MODE'] = 'offline'

trainer = Seq2SeqTrainer(
    model=custom_model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    
)

# %%
trainer.train()

# %%



