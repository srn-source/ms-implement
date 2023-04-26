
import numpy as np
import random
import torch
from functools import cached_property
from datasets import load_dataset
import logging
from itertools import permutations
import copy
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional
logging.basicConfig(level = logging.INFO)
from transformers import  LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
#LLaMATokenizer, LLaMAForCausalLM,
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria
device = "cuda" if torch.cuda.is_available() else "cpu"
import torch.nn.functional as F
import os
from collections import Counter
def seed_every_thing(seed):
    # random.seed(train_seed)
    # np.random.seed(train_seed)
    # torch.manual_seed(train_seed)
    # if torch.cuda.device_count() > 0:
    #     torch.cuda.manual_seed_all(train_seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
def deterministic_random(seed: int) -> random.Random:
    return random.Random(seed)
model1 = {
                "gpt2": "gpt2",
                "stablelm": "stabilityai/stablelm-base-alpha-7b",
                "gpt2-medium": "gpt2-medium",
                "gpt2-large": "gpt2-large",
                "gpt2-xl": "gpt2-xl",
                "gpt_j6b":"EleutherAI/gpt-j-6b",
                "gpt4all_j":"nomic-ai/gpt4all-j",
                #"gpt4all_lora":"nomic-ai/gpt4all-lora",
                "dolly_v2_7b":"databricks/dolly-v2-7b"
                }
model2 = {
                "llama": "decapoda-research/llama-7b-hf",
                "alpaca": "chavinlo/alpaca-native",
                "alpaca-lora": "chainyo/alpaca-lora-7b",
                }
model3 = {
                "gpt3": "text-davinci-002"
                }

def entropy(probs: torch.FloatTensor) -> torch.FloatTensor:
    return -(probs * torch.log2(probs)).nansum()

def to_device(tensor_dict, device):
    return {k: v.to(device) for k, v in tensor_dict.items()}
class stop(StoppingCriteria):
    def __call__(self, iids, _):
        assert iids.shape[0] == 1
        return iids[0][-2:].tolist() == [4906, 25] #"type" , "TYPE" , ":"

class BaseProcessor:
    @cached_property
    def dataset(self):
        return load_dataset(self.dataset_name)
    @cached_property
    def train_split(self):
        if self.dataset_name in ["ag_news"]:
            return self.dataset["train"].map(self.convert_example_to_template_fields)
        else:   
            return self.dataset["train"]
    @cached_property
    def val_split(self):
        return self.dataset["validation"]
    @cached_property
    def test_split(self):
        if self.dataset_name in ["ag_news"]:
            return self.dataset["test"].map(self.convert_example_to_template_fields)
        else:   
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
        tok = RobertaTokenizer.from_pretrained("roberta-large")
        model = RobertaModel.from_pretrained("roberta-large")
    
        logging.info("Start Encoder : {}".format("roberta-large"))
        
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
            
            # nbrs1 = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', n_jobs=-1).fit(emb_train)
            # distances1, indices1 = nbrs.kneighbors(emb_dev)
            
        elif self.kate_metric == "cosine":
            logging.info("Start cosine_similarity...")
            dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
            if self.reversed:
                values, indices = torch.topk(-torch.from_numpy(dist_matrix), k=self.k, dim=-1)
                values_t, indices_t = torch.topk(torch.from_numpy(dist_matrix), k=self.k, dim=-1)
                # print("values == ",values)
                # print("indices == ",indices)
                # print("values_t == ",values_t)
                # print("indices_t == ",indices_t)
            else:
                values, indices = torch.topk(torch.from_numpy(dist_matrix), k=self.k, dim=-1)
                print("values == ",values)
            indices = indices.numpy()
        
        # gggg = []
        # for h in indices:
        #     print("h = ",h)
        #     gggg1 = []
        #     for j in h:
        #         gggg1.append(train_label[j])
        #     gggg.append(gggg1)
        # print("gggg = ",gggg)
        
        
        train_indices_np = np.asarray(train_indices)
        #print(train_indices_np)
        kNN_dev_train = [train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))]
        #print(kNN_dev_train)
        kNN_dev_train = np.concatenate(kNN_dev_train, axis=0).tolist()
        
        return kNN_dev_train
    def make_label_ids(self,tokenizer):
        label_ids = []
        sp = " "
        if self.model_name in model2.keys():
           sp = ""
        for label, label_encoded in zip(
                self.labels,
                tokenizer.batch_encode_plus([ sp + l for l in self.labels])[
                    "input_ids"
                ],
            ):
                print(label, label_encoded)
                label_id = label_encoded[0]
                label_str = tokenizer.convert_ids_to_tokens(label_id)
                if len(label_encoded) > 1:
                    logging.warning(
                        f"Cannot find matching id for {label}, using prefix {label_str}"
                    )
                label_ids.append(label_id)
        
        self.label_ids = torch.tensor(label_ids, dtype=torch.long).to(device)
        logging.info(f"label_ids: {label_ids}")
    def initialize_model1(self,model_name):
        print("initialize_model1 = ",model_name)
        if "/" not in model_name :
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            model.eval().to(device)
            
            return model , tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.eos_token = tokenizer.eos_token
            tokenizer.eos_token_id = tokenizer.eos_token_id
            return model , tokenizer
    def initialize_model2(self,model_name):
        print("initialize_model2 = ",model_name)
        model = LLaMAForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                )
        tokenizer = LLaMATokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        tokenizer.add_bos_token = False
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.eos_token = tokenizer.eos_token
        
        return model , tokenizer
    def initialize_all(self):
        
        model , tokenizer ,model_type  = "", "", ""
        if self.model_name in model1.keys():
            model , tokenizer = self.initialize_model1(model1[self.model_name])
            self.make_label_ids(tokenizer)
            model_type = "model1"
        elif self.model_name in model2.keys(): #please remind that tokenize dont have ""
            model , tokenizer = self.initialize_model2(model2[self.model_name])
            self.make_label_ids(tokenizer)
            model_type = "model2"
        # elif self.model_name in model3.keys():
        #     model , tokenizer = self.initialize_model3(model3[self.model_name])
        #     self.make_label_ids(tokenizer)
        #     model_type = "model3"
            
        return model , tokenizer, model_type
    
    def probe(self,prompt, model , tokenizer, model_type, gen_leng):
        
        decoded = ""
        pred = 1000
        # generation_config = GenerationConfig(
        #                     temperature=0.0,
        #                     top_p=0.95,
        #                     repetition_penalty=1.15,
        #                 )
        if model_type in ["model1", "model2"]:
            # model , tokenizer = self.initialize_model1(model1[self.model_name])
            # self.make_label_ids(tokenizer)
            #print("prompt = ",prompt)
            batch = tokenizer.encode_plus(
            prompt, return_tensors="pt", padding=True
            )
            batch = to_device(batch, device)
            input_length = batch["input_ids"].shape[1]
            if gen_leng > 1:
                with torch.no_grad():
                    output = model.generate(
                        **batch,
                        max_length=input_length + gen_leng,
                        output_hidden_states=True,
                        do_sample=False,
                        output_scores =True,
                        no_repeat_ngram_size=3,
                        temperature= 2.0,
                        return_dict_in_generate =True,
                        stopping_criteria=[stop()],
                        pad_token_id=tokenizer.eos_token_id
                    )
            else:
                with torch.no_grad():
                    output = model.generate(
                        **batch,
                        max_length=input_length + gen_leng,
                        output_hidden_states=True,
                        output_scores =True,
                        do_sample =False,
                        #generation_config=generation_config,
                        return_dict_in_generate =True,
                        pad_token_id=tokenizer.eos_token_id
                    )
            encoded = output.sequences
            decoded = tokenizer.batch_decode(encoded[:, input_length:], skip_special_tokens=True)
            if gen_leng == 1:
                #print("decoded == ",decoded)
                logits_all = output.scores[0]
                for i, raw_completion in enumerate(decoded):
                    logits = logits_all[i,self.label_ids]
                    # print("neg = " , logits_all[i][4633])
                    # print("pos = " , logits_all[i][3967])
                    #print("logits = ",logits)
                    probs= F.softmax(logits, dim=0)
                    pred = probs.argmax(0)
                    #print(pred)
                
            #print("pred = ",pred)
        
            
        assert decoded != ""
        return decoded , pred
    
    def globalentropy_ordering(self,random_train_ids):
        
        model , tokenizer, model_type = self.initialize_all()
        template = "input: {text}\ntype: {label_text}\n\n"
        probe_examples = []
        for perm in tqdm(permutations(random_train_ids),  desc='Subsets', leave = False):
            train_dataset =self.train_split.select(perm)
            
            prompt = ""
            for data_example_in in train_dataset:
                    prompt = prompt + template.format(**data_example_in)
            prompt = prompt  + "input:"
            #print(prompt)
            
            probe_raw , pred = self.probe(prompt, model , tokenizer, model_type , 128)
            #print("probe_raw == ",probe_raw)
            probe_str = probe_raw[0].strip().split("type")[0]
            probe_str = probe_str.strip().split("TYPE")[0]
            probe_str = probe_str.strip().split("Type")[0]
            #print("perm == ",perm)
            print("probe_str == ",probe_str)
            probe_item = self.parse_probe_example(probe_str)
            #print("probe_item == ",probe_item)
            probe_examples.append(probe_item)
            
        prompt = ""
        perm_to_entropy = {}
        for perm in tqdm(permutations(random_train_ids),  desc='Subsets', leave = False):
            
            class_dist = [0] * len(self.labels)
            prompt = self.prompt_start
            #prompts = []
            train_dataset = self.train_split.select(perm)
            
            for data_example in train_dataset:
                prompt = prompt + self.train_template.format(**data_example)
            for data_test in probe_examples:
                prompts123 = prompt + self.eval_template.format(**data_test)
                #print("prompts123 => ",prompts123)
                label_ans , pred= self.probe(prompts123, model , tokenizer, model_type , 1)
                class_dist[pred.item()] += 1
                #prompts.append(prompt + self.eval_template.format(**data_test))
            label_counts = torch.tensor(class_dist)
            #print("label_counts ========== ", label_counts)
            class_distribution = label_counts / label_counts.sum()
            #print("class_distribution ========== ", class_distribution)
            global_entropy = entropy(class_distribution)
            #print("global_entropy ========== ", global_entropy.item())
            perm_to_entropy[perm] = global_entropy.item()
            
        print(perm_to_entropy)
        
        
        best_perm = max(perm_to_entropy.keys(), key=lambda k: perm_to_entropy[k])
        
        print("best_perm ====> ", list(best_perm))
        # print(rtghrthrthrthrth)
        
        return list(best_perm)
    def generate_datasets(self, seed: int):
        logging.info(f"generating datasets using seed {seed}")
        seed_every_thing(self.seed)
        print(self.dataset_name)
        
        
        #self.train_id = random.sample(range(len(self.train_split)), k=self.k)
        
        #self.train_dataset = [self.train_split[i] for i in self.train_id]
        #sp = self.train_split.train_test_split(test_size=self.k, shuffle = True, seed=self.seed , stratify_by_column="label_text",)

        #self.train_dataset = sp["test"]
        
        
        #dataset = load_dataset("my_dataset")
        # for balance dataset
        #train_dataset1, test_dataset1 = train_test_split(self.train_split, test_size=self.k,shuffle = True, random_state=self.seed , stratify=self.train_split["label"])
        #self.train_dataset = Dataset.from_dict(test_dataset1) 
        #print(type(test_dataset1))
        #random.shuffle(self.train_dataset)
        #self.test_dataset = [self.test_split[i] for i in self.test_id]
        #self.test_dataset = [self.test_split[i] for i in range(len(self.test_split))]
        #train_dataset2, test_dataset2 = train_test_split(self.test_split, test_size=20,shuffle = True, random_state=self.seed , stratify=self.test_split["label"])
        #self.test_dataset = Dataset.from_dict(test_dataset2)


        # list_of_k = []
        # balance_k = int(self.k/ len(self.labels))
        # for i in range(len(self.labels)):
        #     start_with_ar = self.train_split.filter(lambda example: example["label"] == i)
            
        #     random_train_ids = random.sample(range(len(start_with_ar)), k= balance_k)
        #     start_with_are =start_with_ar.select(random_train_ids)
        #     for j in range(0,balance_k ):
        #       list_of_k.append(start_with_are[j])
        # random.shuffle(list_of_k)
        # self.train_dataset = Dataset.from_list(list_of_k)


        # list_train_ids = self.kate_process()
        # random_train_ids = self.make_represent(list_train_ids)
        # print("random_train_ids_with_kate = ", random_train_ids)
        
        random_train_ids = deterministic_random(self.seed).sample(range(len(self.train_split)), k=self.k)
        print("random_train_ids = ", random_train_ids)
        
        
        ids_ordering = []
        if self.entropy_ordering:
            ids_ordering = self.globalentropy_ordering(random_train_ids)
            
        print("ids_ordering = ",ids_ordering)
        print("random_train_ids = ", random_train_ids)
        
        
        #print(fghrthrth)
        if len(ids_ordering) > 0:
            random_train_ids = ids_ordering
        self.train_dataset =self.train_split.select(random_train_ids)
        #print("random_train_ids ==> ",random_train_ids)

        random_test_ids = deterministic_random(42).sample(range(len(self.test_split)), k=1000)
        self.test_dataset = self.test_split.select(random_test_ids)
        #self.test_dataset = [self.test_split[i] for i in range(len(self.test_split))]
        # if self.dataset_name in ["ag_news"]:
        #     self.train_dataset = self.train_dataset.map(self.convert_example_to_template_fields)
        #     self.test_dataset = self.test_dataset.map(self.convert_example_to_template_fields)
    def make_represent(self,list_train_ids):
        ooo = []
        for i in range(0,self.k):
            counts = Counter(lst[i] for lst in list_train_ids)
            max_count = min(counts.values())
            output = [k for k, v in counts.items() if v == max_count][0]
            ooo.append(output)
        return ooo
        
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
        prompts_cali = []
        prompts_cali2 = []
        prompts_cali3 = []
        
        label_test = []
        prompt = self.prompt_start
        cali = {"text": "N/A"}
        cali2 = {"text": "[MASK]"}
        cali3 = {"text": ""}
        
        if self.kate:
            list_train_ids = self.kate_process()
            new_list = self.make_represent(list_train_ids)
            
            print("kate id = ", list_train_ids)
            for data_example , data_test in zip(list_train_ids , self.test_dataset):
                prompt = self.prompt_start
                
                # print("before shuffle = " , data_example)
                # random.shuffle(data_example)
                # print("after shuffle = " , data_example)
                
                train_dataset_info = [self.train_split[i]for i in data_example]
                # print(type(train_dataset_info))
                # print(train_dataset_info)
                # #train_dataset_info= Dataset.from_dict(train_dataset_info) 
                # if self.dataset_name in ["ag_news"]:
                #     self.test_dataset = self.test_dataset.map(self.convert_example_to_template_fields)

                for data_example_in in train_dataset_info:
                    #print(data_example_in)
                    prompt = prompt + self.train_template.format(**data_example_in)
                    
                prompts.append(prompt + self.eval_template.format(**data_test))
                prompts_cali.append(prompt + self.eval_template.format(**cali))
                prompts_cali2.append(prompt + self.eval_template.format(**cali2))
                prompts_cali3.append(prompt + self.eval_template.format(**cali3))
                label_test.append(data_test["label_text"].strip())

            
        else:
            # print("before shuffle = " , self.train_dataset)
            # random.shuffle(self.train_dataset)
            # print("after shuffle = " , self.train_dataset)
            for data_example in self.train_dataset:
                prompt = prompt + self.train_template.format(**data_example)
            for data_test in self.test_dataset:
                prompts.append(prompt + self.eval_template.format(**data_test))
                prompts_cali.append(prompt + self.eval_template.format(**cali))
                prompts_cali2.append(prompt + self.eval_template.format(**cali2))
                prompts_cali3.append(prompt + self.eval_template.format(**cali3))
                
                label_test.append(data_test["label_text"].strip())
                
                
                
        test_kwargs = {
           "labels": self.labels,
           "label_test": label_test,
           #"labels_token_gpt3":self.labels_token_gpt3
        }
        self.model_kwargs.update(test_kwargs)
        
        # print(prompts_cali[0])
        # print("==============================")
        # print(prompts[0])
        # print("==============================")
       
        return prompts , prompts_cali , prompts_cali2 , prompts_cali3
        
class SST2Processor(BaseProcessor):
    #def __init__(self, seed , k , kate, kate_metric, reversed, model_class) :
    def __init__(self, seed: int = 87 , k: int = 8 , kate: bool = False, kate_metric: str = "euclidean" , reversed: bool =False ,model_name : str = "" , entropy_ordering: bool =False ) :
        
        self.k = k
        self.kate = kate
        self.seed = seed
        self.kate_metric = kate_metric
        self.reversed = reversed
        self.entropy_ordering = entropy_ordering
        self.dataset_name = "SetFit/sst2"
        self.prompt_start = "Below is couple of movie reviews and their corresponding sentiments. Write a sentiment that appropriately completes the request.\n\n"
        #self.prompt_start = ""
        self.train_template = "Review: {text}\n" "Sentiment: {label_text}\n\n"
        if model_name in model2.keys(): 
            self.eval_template = "Review: {text}\n" "Sentiment: "
        else:
            self.eval_template = "Review: {text}\n" "Sentiment:"

        self.model_name = model_name
        self.labels = ["negative", "positive"]
        #https://platform.openai.com/tokenizer?view=bpe
        #self.labels_token_gpt3 = [4633, 3967]
        self.model_kwargs = {"labels": self.labels }
        self.generate_datasets(seed)
    def parse_probe_example(self, s: str):
        return {"text": s, "label_text": "positive"}
        

class AGNewsProcessor(BaseProcessor):
    def __init__(self, seed: int = 87 , k: int = 8 , kate: bool = False, kate_metric: str = "euclidean" , reversed: bool =False, model_name : str = "" , entropy_ordering: bool =False) :
        
        
        self.k = k
        self.kate = kate
        self.seed = seed
        self.kate_metric = kate_metric
        self.entropy_ordering = entropy_ordering
        self.reversed = reversed
        self.dataset_name = "ag_news"
        self.prompt_start = "Below is couple of news article and their corresponding answer. Write an answer that appropriately completes the request.\n\n"
        #self.prompt_start = ""
        self.train_template = "Article: {text}\n" "Answer: {label_text}\n\n"
        if model_name in model2.keys():
          self.eval_template = "Article: {text}\n" "Answer: "
        else:
          self.eval_template = "Article: {text}\n" "Answer:"
        self.model_name = model_name
        self.labels = ["World", "Sports", "Business", "Technology"]
        self.model_kwargs = {"labels": self.labels }
        self.generate_datasets(seed)
        
    def convert_example_to_template_fields(self, example: Dict):
            label_text = self.labels[example["label"]]
            return {"text": example["text"], "label_text": label_text}
    def parse_probe_example(self, s: str):
        return {"text": s,"label_text": "World"}