
import numpy as np
import random
import torch
from functools import cached_property
from datasets import load_dataset
import logging
import copy
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise

logging.basicConfig(level = logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
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
        tok = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base")
    
        logging.info("Start Encoder : {}".format("roberta-base"))
        
        test_text = []
        test_label = []
        
        for key in self.test_split:
            text = key['text'].strip()
            label = int(key['label'])
            test_text.append(text)
            test_label.append(label)
            
        train_text = []
        train_label = []
        
        for key in self.train_split:
            text = key['text'].strip()
            label = int(key['label'])
            train_text.append(text)
            train_label.append(label)
            
        train_indices = list(range(len(train_text)))
        
        corpus = test_text + train_text
        n_dev = len(test_label)
        n_train = len(train_text)
        
        model.to("cuda")
        X = self.decode(tok, model, corpus)
        emb_train = X[n_dev:]
        emb_dev = X[:n_dev]

        logging.info("n_dev = {} n_train = {} all corpus = {}".format(n_dev,n_train,len(corpus)))
        # logging.info("Start NearestNeighbors...")
        # nbrs = NearestNeighbors(n_neighbors=(self.k), algorithm='ball_tree', n_jobs=-1).fit(emb_train)
        # distances, indices = nbrs.kneighbors(emb_dev)
        
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
        #print(train_indices_np)
        kNN_dev_train = [train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))]
        #print(kNN_dev_train)
        kNN_dev_train = np.concatenate(kNN_dev_train, axis=0).tolist()
        
        return kNN_dev_train
    
    def generate_datasets(self, seed: int):
        logging.info(f"generating datasets using seed {seed}")
        print(self.dataset_name)
        
        self.train_id = random.sample(range(len(self.train_split)), k=self.k)
        #self.test_id = random.sample(range(len(self.test_split)), k=2)
        
        self.train_dataset = [self.train_split[i] for i in self.train_id]
        #self.test_dataset = [self.test_split[i] for i in self.test_id]
        self.test_dataset = [self.test_split[i] for i in range(len(self.test_split))]
    
    def create_prompt(self, model_name):
        
        # if model_name =="llama":
        #     self.labels = ["neg", "pos"]
            
        #     for t in self.train_dataset:
        #         for key, value in t.items():
        #             if key == "label_text":
        #                 t[key] = value[:3]
                        
        #     for t in self.test_dataset:
        #         for key, value in t.items():
        #             if key == "label_text":
        #                 t[key] = value[:3]
        
        prompts = []
        label_test = []
        prompt = self.prompt_start
        
        if self.kate:
            list_train_ids = self.kate_process()
            
            for data_example , data_test in zip(list_train_ids , self.test_dataset):
                prompt = self.prompt_start
                
                train_dataset_info = [self.train_split[i] for i in data_example]
                
                for data_example_in in train_dataset_info:
                    #print(data_example_in)
                    prompt = prompt + self.train_template.format(**data_example_in)
                    
                prompts.append(prompt + self.eval_template.format(**data_test))
                label_test.append(data_test["label_text"].strip())
                
        else:
            
            for data_example in self.train_dataset:
                prompt = prompt + self.train_template.format(**data_example)
            for data_test in self.test_dataset:
                prompts.append(prompt + self.eval_template.format(**data_test))
                label_test.append(data_test["label_text"].strip())
                
                
        test_kwargs = {
           "labels": self.labels,
            "label_test": label_test,
        }
        self.model_kwargs.update(test_kwargs)
        
        print(prompts[0])
        print("==============================")
        print(prompts[1])
        return prompts
        
class SST2Processor(BaseProcessor):
    def __init__(self, seed: int = 87 , k: int = 4 , kate: bool = False, kate_metric: str = "euclidean" , reversed: bool =False) :
        
        self.k = k
        self.kate = kate
        self.kate_metric = kate_metric
        self.reversed = reversed
        self.dataset_name = "SetFit/sst2"
        self.prompt_start = "Below is couple of movie reviews and their corresponding sentiments. Write a sentiment that appropriately completes the request.\n\n"
        self.train_template = "Review: {text}\n" "Sentiment: {label_text}\n\n"
        self.eval_template = "Review: {text}\n" "Sentiment:"
        
        self.labels = ["negative", "positive"]
        self.model_kwargs = {"labels": self.labels }
        self.generate_datasets(seed)