from datasets import load_dataset, concatenate_datasets
from data.data_utils.utils import preprocess_function

class DataPreparation:
    def __init__(self, dataset_paths: dict, tokenizer, bidirectional=False, invert=False,src_lang=None, tgt_lang=None) -> None:
        self.dataset_paths = dataset_paths
        self.tokenizer = tokenizer
        self.bidirectional = bidirectional
        self.invert = invert
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        assert not(invert or bidirectional), "Invert and Bidirectional cannot be true at the same time"
        assert (self.src_lang is not None), "src_lang cannot be None"
        assert (self.tgt_lang is not None), "tgt_lang cannot be None"


    def load_data(self):
        self.dataset = load_dataset("json", data_files=self.dataset_paths)

    def preprocess_data(self):
        self.preprocessed_dataset = self.dataset.map(
            preprocess_function, 
            fn_kwargs={'tokenizer': self.tokenizer}, 
            batched=True
        )
        self.train_dataset = self.preprocessed_dataset['train']
        self.test_dataset = self.preprocessed_dataset['test']
        self.val_dataset = self.preprocessed_dataset['val']
        self.bidirectional_mix()

    def mix_data(self):
        pass

    def invert_data(self):
        if self.invert:
            pass


    def bidirectional_mix(self):
        if self.bidirectional:
            self.tokenizer.src_lang = self.tgt_lang
            self.tokenizer.tgt_lang = self.src_lang
            bi_dataset = self.dataset.map(
                self.preprocess_function,
                fn_kwargs={'tokenizer': self.tokenizer, 'input_col': 'target', 'target_col': 'input'},
                batched=True
            )
            self.train_dataset = concatenate_datasets([self.train_dataset, bi_dataset['train']])
            self.test_dataset = concatenate_datasets([self.test_dataset, bi_dataset['test']])
            self.val_dataset = concatenate_datasets([self.val_dataset, bi_dataset['val']])
