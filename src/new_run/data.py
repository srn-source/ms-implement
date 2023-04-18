
import numpy as np
import random
import torch
from functools import cached_property
from datasets import load_dataset
import logging
import copy

logging.basicConfig(level = logging.INFO)


class BaseProcessor:
    @cached_property
    def dataset(self):
        return load_dataset(self.dataset_name)
    @cached_property
    def train_split(self):
        return self.dataset["train"]
    @cached_property
    def val_split(self):
        return self.dataset["validation"]
    @cached_property
    def test_split(self):
        return self.dataset["test"]
    def generate_datasets(self, seed: int):
        logging.info(f"generating datasets using seed {seed}")
        print(self.dataset_name)
        self.train_id = random.sample(range(len(self.train_split)), k=self.k)
        #self.test_id = random.sample(range(len(self.test_split)), k=500)
        
        self.train_dataset = [self.train_split[i] for i in self.train_id]
        self.test_dataset = [self.test_split[i] for i in range(len(self.test_split))]
        
    def create_prompt(self):
        prompts = []
        label_test = []
        prompt = self.prompt_start
        for data_example in self.train_dataset:
            prompt = prompt + self.train_template.format(**data_example)
        #print(prompt)
        
        for data_test in self.test_dataset:
            prompts.append(prompt + self.eval_template.format(**data_test))
            label_test.append(data_test["label_text"].strip())
        
        test_kwargs = {
           "labels": self.labels,
            "label_test": label_test,
        }
        self.model_kwargs.update(test_kwargs)
        
        #print(prompts[0:2])
        return prompts
        
class SST2Processor(BaseProcessor):
    def __init__(self, seed: int = 87 , k: int = 4):
        
        self.k = k
        self.dataset_name = "SetFit/sst2"
        self.prompt_start = "Below is couple of movie reviews and their corresponding sentiments. Write a sentiment that appropriately completes the request.\n\n"
        self.train_template = "Review: {text}\n" "Sentiment: {label_text}\n\n"
        self.eval_template = "Review: {text}\n" "Sentiment:"
        
        self.labels = ["negative", "positive"]
        self.model_kwargs = {"labels": self.labels }
        self.generate_datasets(seed)