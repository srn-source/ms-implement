from datasets import load_dataset

import random
import torch
import numpy as np
from typing import List
import logging


logging.basicConfig(level = logging.INFO)

class SST2Processor():
    def __init__(self,  k, seed, dataset, tokenizer):
        self.dataset_name = "SetFit/sst2"
        self.tokenizer = tokenizer
        self.seed = seed
        self.k = k
        dataset1 = load_dataset(self.dataset_name)
        self.train_split = dataset1["train"]
        self.test_split = dataset1["test"]
        self.val_split = dataset1["validation"]
        
        self.train_id = random.sample(range(len(self.train_split)), k=self.k)
        # self.template = template
        # self.tmp_idx = tmp_idx
        #self.mode = mode
    def generate_setOfDemon(self,template):
        
        label_words = ["terrible", "great"]
        demonstrations = []
        
        train_id = self.train_id
        logging.info(f"{train_id}")
        few_train = [self.train_split[i] for i in train_id]
        logging.info("few_train = {}".format(len(few_train)))
        
        for key in few_train:
            text = key['text'].strip()
            label = int(key['label'])
            
            tokens_input = self.tokenizer(text)["input_ids"] 
            tokens_label = self.tokenizer(" " + (template % label_words[label]))["input_ids"] 
            demonstrations = demonstrations + tokens_input + tokens_label
            #demonstrations_t = demonstrations_t+ text+ " " + (template % label_words[label])
            
        return demonstrations
    
    def generate_setOfDemon_channel(self,template):
        
        label_words = ["terrible", "great"]
        demonstrations = []
        
        train_id = self.train_id
        logging.info(f"{train_id}")
        few_train = [self.train_split[i] for i in train_id]
        logging.info("few_train = {}".format(len(few_train)))
        
        for key in few_train:
            text = key['text'].strip()
            label = int(key['label'])
            
            p = (template % label_words[label])
            if len(demonstrations)>0:
                p = " " + p
            
            tokens_input = self.tokenizer(text)["input_ids"] 
            tokens_label = self.tokenizer(p)["input_ids"] 
            demonstrations = demonstrations + tokens_label + tokens_input
            #demonstrations_t = demonstrations_t+ text+ " " + (template % label_words[label])
            
        return demonstrations
    
    def get_prompts(self , few_train,test_data):
        label_words = ["terrible", "great"]
        
        #print(few_train)
        demonstrations = []
        demonstrations_t = ""
        for key in few_train:
            text = key['text'].strip()
            label = int(key['label'])
            
            # if len(demonstrations)>0:
            #     text = " " + text
            
            tokens_input = self.tokenizer(text)["input_ids"] 
            tokens_label = self.tokenizer(" " + (self.template % label_words[label]))["input_ids"] 
            demonstrations = demonstrations + tokens_input + tokens_label
            demonstrations_t = demonstrations_t+ text+ " " + (self.template % label_words[label])
        
        #test_inputs = [self.tokenizer(sent['text'].strip())["input_ids"] for sent in test_data]
        # data = []
        # for key in test_data:
        #     text = key['text'].strip()
        #     label = int(key['label'])
        #     data.append((text, label))
        # print(demonstrations)
        # print(demonstrations_t)
        # print(test_inputs[0])
        
        return demonstrations 
    def generate_datasets2212(self):
        assert self.k < 8
        logging.info(f"generating datasets using seed = {self.seed}, k = {self.k} , Dataset = {self.dataset_name}")
        
        dataset1 = load_dataset(self.dataset_name)
        train_split = dataset1["train"]
        test_split = dataset1["test"]
        val_split = dataset1["validation"]
        logging.info("train_split = {} test_split = {} val_split = {}".format(len(train_split),len(test_split),len(val_split)))


        train_id = [10,55,415,45]#random.sample(range(len(train_split)), k=self.k)
        logging.info(f"{train_id}")
        
        # test_id = random.sample(range(len(test_split)), k=self.k)
        # logging.info(f"{test_id}")
        
        few_train = [train_split[i] for i in train_id]
        logging.info("few_train = {}".format(len(few_train)))
        
        # few_test = [test_split[i] for i in test_id]
        # logging.info("few_test = {}".format(len(few_test)))
        
        demonstrations = self.get_prompts(few_train,test_split)
        return demonstrations
    
    def generate_test_set(self):
        #assert self.mode == "Test"
        data = []
        data_token = []
        data_token_with_space = []
        #dataset1 = load_dataset(self.dataset_name)
        #test_split = dataset1["test"]
        #j = 0
        #[self.tokenizer(sent['text'].strip())["input_ids"] for sent in test_data]
        for key in self.test_split:
            text = key['text'].strip()
            label = int(key['label'])
            data.append((text, label))
            data_token.append(self.tokenizer(text)["input_ids"])
            data_token_with_space.append(self.tokenizer(" ",text)["input_ids"])
            # j = j+1
            # if j >6:
            #     break
        return data , data_token , data_token_with_space