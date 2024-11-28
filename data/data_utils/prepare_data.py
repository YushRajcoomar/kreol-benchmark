import logging
import random

from datasets import load_dataset, concatenate_datasets
from .utils import preprocess_function
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, dataset_paths: dict, tokenizer, bidirectional=False, invert=False, src_lang=None, tgt_lang=None, rating_adapter = False) -> None:
        self.dataset_paths = dataset_paths
        self.tokenizer = tokenizer
        self.bidirectional = bidirectional
        self.invert = invert
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.rating_adapter = rating_adapter

        assert not(invert and bidirectional), "Invert and Bidirectional cannot be true at the same time"
        assert (self.src_lang is not None), "src_lang cannot be None"
        assert (self.tgt_lang is not None), "tgt_lang cannot be None"

    def load_data(self):
        logger.info("Loading data from paths: %s", self.dataset_paths)
        self.dataset = load_dataset("json", data_files=self.dataset_paths)
        logger.info("Data loaded successfully. Dataset keys: %s", self.dataset.keys())
        logger.info(f"Dataset lengths: {self.dataset.num_rows}")

    def preprocess_data(self):
        logger.info("Preprocessing data...")
        self.preprocessed_dataset = self.dataset.map(
            preprocess_function, 
            fn_kwargs={'tokenizer': self.tokenizer}, 
            batched=True
        )
        logger.info("Data preprocessed successfully.")
        self.train_dataset = self.preprocessed_dataset['train']
        logger.info("Train dataset size after preprocessing: %d", len(self.train_dataset))
        if 'test' in self.preprocessed_dataset:
            self.test_dataset = self.preprocessed_dataset['test']
            logger.info("Test dataset size after preprocessing: %d", len(self.test_dataset))
        if 'val' in self.preprocessed_dataset:
            self.val_dataset = self.preprocessed_dataset['val']
            logger.info("Validation dataset size after preprocessing: %d", len(self.val_dataset))
        self.bidirectional_mix(self.dataset)
        
        

    def _load_mixer_data(self, mixer_dict):
        datasets_list = []
        for key, (path, ratio) in mixer_dict.items():
            dataset = load_dataset('json', data_files=path)['train']
            dataset = dataset.shuffle(seed=42)
            sampled_dataset = dataset.select(range(int(len(dataset) * ratio)))

            if self.rating_adapter:
                sampled_dataset = self.add_rating(sampled_dataset)

            datasets_list.append(sampled_dataset)   

        concatenated_dataset = concatenate_datasets(datasets_list)
        
        return concatenated_dataset

    def add_rating(self,dataset,rating_default=5):
        # rating set to random but theoretically does not matter
        if 'rating' not in dataset.features:
            dataset = dataset.add_column('rating',[rating_default for i in range(len(dataset))])
        if 'predicted_text' not in dataset.features:
            dataset = dataset.add_column('predicted_text', dataset['target'])
        return dataset


    def mix_data(self, mixer_dict: str, train_only=False):
        new_dataset = self._load_mixer_data(mixer_dict)
        logger.info("New data loaded successfully. Number of examples: %d", len(new_dataset))

        preprocessed_new_data = new_dataset.map(
            preprocess_function, 
            fn_kwargs={'tokenizer': self.tokenizer}, 
            batched=True
        )
        logger.info("New data preprocessed successfully. Number of examples: %d", len(preprocessed_new_data))
        
        self.train_dataset = concatenate_datasets([self.train_dataset, preprocessed_new_data])

        if self.bidirectional:
            self.bidirectional_mix(new_dataset,train_only=train_only)
            logger.info("Bidirectional data preprocessed successfully.")

        if self.rating_adapter:
            self.train_dataset = self.add_rating(self.train_dataset)
            logger.info("Train dataset size after mixing: %d", len(self.train_dataset))
            if 'test' in self.preprocessed_dataset:
                self.test_dataset = self.add_rating(self.test_dataset)
                logger.info("Test dataset size after mixing: %d", len(self.test_dataset))
            if 'val' in self.preprocessed_dataset:
                self.val_dataset = self.add_rating(self.val_dataset)
                logger.info("Validation dataset size after mixing: %d", len(self.val_dataset))
        
        self.train_dataset=self.train_dataset.shuffle(seed=42)
        logger.info("Data mixing complete.")

    # def _mix_dataset(self, original_dataset, new_dataset, mix_percentage):
    #     if mix_percentage <= 0 or len(new_dataset) == 0:
    #         return original_dataset
        
    #     num_to_sample = int(len(original_dataset) * mix_percentage)
    #     num_to_sample = min(len(new_dataset),num_to_sample)
    #     logger.info("Sampling %d examples from new dataset for mixing.", num_to_sample)

    #     sampled_new_data = new_dataset.shuffle(seed=42).select(range(num_to_sample))
    #     mixed_dataset = concatenate_datasets([original_dataset, sampled_new_data])
        
    #     logger.info("Mixed dataset size before final shuffling: %d", len(mixed_dataset))
    #     return mixed_dataset.shuffle(seed=42)

    def invert_data(self):
        if self.invert:
            logger.info("Inverting data...")
            inverted_datasets = self.dataset.map(
                preprocess_function,
                fn_kwargs={'tokenizer': self.tokenizer, 'input_col': 'target', 'target_col': 'input'},
                batched=True
            )
            self.train_dataset = inverted_datasets['train']
            self.test_dataset = inverted_datasets['test']
            self.val_dataset = inverted_datasets['val']
            logger.info("Data inversion complete.")

    def bidirectional_mix(self,dataset,train_only=False):
        if self.bidirectional:
            logger.info("Creating bidirectional mix...")
            self.tokenizer.src_lang = self.tgt_lang
            self.tokenizer.tgt_lang = self.src_lang
            bi_dataset = dataset.map(
                preprocess_function,
                fn_kwargs={'tokenizer': self.tokenizer, 'input_col': 'target', 'target_col': 'input'},
                batched=True
            )
            logger.info("Bidirectional preprocessing complete.")

            if not train_only:
                self.train_dataset = concatenate_datasets([self.train_dataset, bi_dataset['train']])
                self.test_dataset = concatenate_datasets([self.test_dataset, bi_dataset['test']])
                self.val_dataset = concatenate_datasets([self.val_dataset, bi_dataset['val']])
            else:
                self.train_dataset = concatenate_datasets([self.train_dataset, bi_dataset])
            logger.info("Bidirectional mix completed.")

# # Example usage

# checkpoint = "/home/yush/kreol-benchmark/checkpoint_tests/checkpoint-11_best500ft"  # specific model path
# tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="en_XX", tgt_lang="cr_CR")

# data_preparation = DataPreparation(
#     dataset_paths={
#         'train':'/home/yush/kreol-benchmark/data_collection/notebooks/cr_en_dict_definitions.json',
#         'test':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_test.jsonl',
#         'val':'/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_dev.jsonl'},
#     tokenizer=tokenizer,
#     bidirectional=True,
#     src_lang="en_XX",
#     tgt_lang="cr_CR",
#     rating_adapter=True
# )

# mixer_dict = {
#     'hq500':['/home/yush/kreol-benchmark/data/lang_data/en-cr/hq_sentences/cr_en_hq_gptapi_500.json', 0.3],
#     'dict':['/home/yush/kreol-benchmark/data_collection/notebooks/cr_en_dict_sentences.json', 0.3],
#     'train_base':['/home/yush/kreol-benchmark/data/lang_data/en-cr/en-cr_train.jsonl', 0.05],
# }

# data_preparation.load_data()
# data_preparation.preprocess_data()
# data_preparation.mix_data(mixer_dict)

# #testing
# print(f"Train dataset size: {len(data_preparation.train_dataset)}")
# print(f"Test dataset size: {len(data_preparation.test_dataset)}")
# print(f"Validation dataset size: {len(data_preparation.val_dataset)}")
# # print("Sample from train dataset:", data_preparation.train_dataset[0])