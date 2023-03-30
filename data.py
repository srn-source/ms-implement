from datasets import load_dataset

import random
import torch
import numpy as np
from typing import List
import logging
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from transformers import GPT2Tokenizer, GPT2Model
logging.basicConfig(level = logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SST2Processor():
    def __init__(self,  k, seed, dataset, tokenizer, kate_metric,reversedCosi , encoder_kate, use_calibration, ensemble):
        self.dataset_name = dataset
        self.tokenizer = tokenizer
        self.seed = seed
        self.k = k
        dataset1 = load_dataset(self.dataset_name)
        self.train_split = dataset1["train"]
        self.test_split = dataset1["test"]
        #self.val_split = dataset1["validation"]
        self.kate_metric = kate_metric
        self.reversed = reversedCosi
        self.encoder_kate = encoder_kate
        self.use_calibration = use_calibration
        self.ensemble = ensemble
        
        self.train_id = random.sample(range(len(self.train_split)), k=self.k)
        # self.template = template
        # self.tmp_idx = tmp_idx
        #self.mode = mode
    def generate_setOfDemon(self,template, group_id_kate = []):
        
        label_words = ["terrible", "great"]
        demonstrations = []
        
        train_id = self.train_id if len(group_id_kate) == 0 else group_id_kate
        
        #logging.info(f"{train_id}")
        few_train = [self.train_split[i] for i in train_id]
        #logging.info("few_train = {}".format(len(few_train)))
        
        for key in few_train:
            text = key['text'].strip()
            label = int(key['label'])
            
            if text[-1] != ".":
                text = text + " ."
            
            tokens_input = self.tokenizer(text)["input_ids"] 
            tokens_label = self.tokenizer(" " + (template % label_words[label]))["input_ids"] 
            demonstrations = demonstrations + tokens_input + tokens_label
            #demonstrations_t = demonstrations_t+ text+ " " + (template % label_words[label])
            
        return demonstrations
    
    def generate_setOfDemon_channel(self,template , group_id_kate = []):
        
        label_words = ["terrible", "great"]
        demonstrations = []
        
        train_id = self.train_id if len(group_id_kate) == 0 else group_id_kate
        #logging.info(f"{train_id}")
        few_train = [self.train_split[i] for i in train_id]
        #logging.info("few_train = {}".format(len(few_train)))
        
        for key in few_train:
            text = key['text'].strip()
            label = int(key['label'])
            
            if text[-1] != ".":
                text = text + " ."
                
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
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    def decode(self, tok, model, corpus ):
        embeddings = []
        
        for corpus_tmp in tqdm(self.chunks(corpus, 128)):
            encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
            sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
            sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(device), torch.LongTensor(attn_mask).to(device)

            with torch.no_grad():
                embedding_output_batch = model(sentence_batch, attn_mask)            
                sentence_embeddings = embedding_output_batch[0][:, 0, :]
            embeddings.append(sentence_embeddings.detach().cpu().numpy())
                
    
        return np.concatenate(embeddings, axis=0)

    def kate_process(self):
        #corpus = dev_corpus + train_corpus
        tok = RobertaTokenizer.from_pretrained(self.encoder_kate)
        model = RobertaModel.from_pretrained(self.encoder_kate)
    
        logging.info("Start Encoder : {}".format(self.encoder_kate))
         
        test_text = []
        test_label = []
        c = 0
        for key in self.test_split:
            text = key['text'].strip()
            label = int(key['label'])
            test_text.append(text)
            test_label.append(label)
            c =c + 1
            # if c > 200:
            #     break
        print("len test_split = " , c)
        train_text = []
        train_label = []
        c = 0
        for key in self.train_split:
            text = key['text'].strip()
            label = int(key['label'])
            train_text.append(text)
            train_label.append(label)
            c =c + 1
            # if c > 200:
            #     break
        print("len train_split = " , c)
        train_indices = list(range(len(train_text)))
        
        corpus = test_text + train_text
        n_dev = len(test_label)
        n_train = len(train_text)
        model.to(device)
        X = self.decode(tok, model, corpus)
        emb_train = X[n_dev:]
        emb_dev = X[:n_dev]

        logging.info("n_dev = {} n_train = {} all corpus = {}".format(n_dev,n_train,len(corpus)))
        #print("emb_train = ", emb_train)
        print("emb_train len= ", len(emb_train))
        
        
        if self.kate_metric == "euclidean":
            logging.info("Start NearestNeighbors...")
            nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', n_jobs=-1).fit(emb_train)
            distances, indices = nbrs.kneighbors(emb_dev)
        elif self.kate_metric == "cosine":
            logging.info("Start cosine_similarity...")
            dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
            if self.reversed:
                values, indices = torch.topk(-torch.from_numpy(dist_matrix), k=self.k, dim=-1)
            else:
                values, indices = torch.topk(torch.from_numpy(dist_matrix), k=self.k, dim=-1)
            indices = indices.numpy()
        
        train_indices_np = np.asarray(train_indices)
        #print("train_indices_np = ",train_indices_np)
        kNN_dev_train = [train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))]
        #print("kNN_dev_train = ",kNN_dev_train)
        kNN_dev_train = np.concatenate(kNN_dev_train, axis=0).tolist()
        #print("kNN_dev_train = ",kNN_dev_train)
        #print("kNN_dev_train = ",kNN_dev_train)
        
        
        
        #prompt = [demonstrations.copy() + test_input + prefix[:tmp_idx] for test_input in test_inputs_token]
        
        return kNN_dev_train
    
    def kate_generate_promt(self, group_id_kate , test_inputs_token , prefix , template):
        prompt = []
        prompt_calibration = []
        na_token = self.tokenizer("N/A")["input_ids"]
        label_words = ["terrible", "great"]
        
        for test_input, group_id in zip(test_inputs_token, group_id_kate):
            #print("group_id = ",group_id)
            if self.ensemble:
                for g_id in group_id:
                    #prompt1 = demonstrations.copy() + test_input + prefix
                    
                    few_train = self.train_split[g_id]
                    text = few_train['text'].strip()
                    label = int(few_train['label'])
                    
                    if text[-1] != ".":
                        text = text + " ."
                        
                    tokens_input = self.tokenizer(text)["input_ids"] 
                    tokens_label = self.tokenizer(" " + (template % label_words[label]))["input_ids"] 
                    
                    prompt1 = tokens_input + tokens_label + test_input + prefix
                    prompt2 = tokens_input + tokens_label + na_token + prefix
                    
                    prompt.append(prompt1)
                    prompt_calibration.append(prompt2)
            else:
                demonstrations = self.generate_setOfDemon(template , group_id)
                prompt1 = demonstrations.copy() + test_input + prefix
                prompt2 = demonstrations.copy() + na_token + prefix
                prompt.append(prompt1)
                prompt_calibration.append(prompt2)
                
        return prompt , prompt_calibration
    
    def ensemble_generate_promt(self, test_inputs_token , prefix , template):
        prompt = []
        prompt_calibration = []
        na_token = self.tokenizer("N/A")["input_ids"]
        label_words = ["terrible", "great"]
        
        few_train = [self.train_split[i] for i in self.train_id]
        for test_input in test_inputs_token:
            for key in few_train:
                text = key['text'].strip()
                label = int(key['label'])
                
                if text[-1] != ".":
                    text = text + " ."
                            
                tokens_input = self.tokenizer(text)["input_ids"] 
                tokens_label = self.tokenizer(" " + (template % label_words[label]))["input_ids"] 
                        
                prompt1 = tokens_input + tokens_label + test_input + prefix
                prompt2 = tokens_input + tokens_label + na_token + prefix
                        
                prompt.append(prompt1)
                prompt_calibration.append(prompt2)
            
        return prompt , prompt_calibration
    
    
    def kate_generate_promt_channel(self, group_id_kate , data_token_with_space , prefix , template):
        
        #prompt = [demonstrations.copy() + prefix for test_input in data_token_with_space]
        prompt = []
        label_ensemble = []
        label_words = ["terrible", "great"]
        for test_input, group_id in zip(data_token_with_space, group_id_kate):
            #print("group_id = ",group_id)
            if self.ensemble:
                for g_id in group_id:
                    
                    few_train = self.train_split[g_id]
                    text = few_train['text'].strip()
                    label = int(few_train['label'])
                        
                    if text[-1] != ".":
                        text = text + " ."
                            
                    p = (template % label_words[label])
                    
                    tokens_input = self.tokenizer(text)["input_ids"] 
                    tokens_label = self.tokenizer(p)["input_ids"] 
                    prompt1 = tokens_label + tokens_input + prefix
                    
                    
                    prompt.append(prompt1)
                    label_ensemble.append(test_input)
            else:
                demonstrations = self.generate_setOfDemon_channel(template , group_id)
                prompt1 = demonstrations.copy() + prefix
                
                prompt.append(prompt1)
        return prompt , label_ensemble