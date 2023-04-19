import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import copy
from typing import Dict, List, Optional
import torch.nn.functional as F
# from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

from tqdm import tqdm

logging.basicConfig(level = logging.INFO)


MODELS_gen_hf = {
          "gpt2": "gpt2",
          "gpt2-medium": "gpt2-medium",
          "gpt2-large": "gpt2-large",
          "gpt_j6b":"EleutherAI/gpt-j-6b",
          "gpt4all_j":"nomic-ai/gpt4all-j",
          #"gpt4all_lora":"nomic-ai/gpt4all-lora",
          "dolly_v2_7b":"databricks/dolly-v2-7b"
          }


def to_device(tensor_dict, device):
    return {k: v.to(device) for k, v in tensor_dict.items()}

class GPT2Wrapper:
    def initialize_model(cls, model_name):
        if "/" not in model_name :
            return AutoModelForCausalLM.from_pretrained(model_name)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                )
    def __init__(
        self,
        model_name: str,
        
        batch_size: int = 8,
        k: int = 4,
        use_calibration: bool = False,
        labels: List[str] = None,
        label_test: List[str] = None,
        
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logging.warning(f"Cannot find gpu, setting device to cpu.")
        self.batch_size = batch_size
        #self.calibrate = calibrate
        logging.info(f"Setting batch_size={batch_size}")

        model_hf = MODELS_gen_hf[model_name]
        self.use_calibration = use_calibration
        self.tokenizer = AutoTokenizer.from_pretrained(model_hf)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_test = label_test

        
        
        logging.info(f"Initializing {model_name}")
        self.model_name = model_name
        self.model = self.initialize_model(model_hf)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        for param in self.model.parameters():
            param.requires_grad = False

        if "/" not in model_hf :
            self.model.eval().to(self.device)
        
        label_ids = []
        if labels is not None:
            for label, label_encoded in zip(
                labels,
                self.tokenizer.batch_encode_plus([" " + l for l in labels])[
                    "input_ids"
                ],
            ):
                print(label, label_encoded)
                label_id = label_encoded[0]
                label_str = self.tokenizer.convert_ids_to_tokens(label_id)
                if len(label_encoded) > 1:
                    logging.warning(
                        f"Cannot find matching id for {label}, using prefix {label_str}"
                    )
                label_ids.append(label_id)

        self.labels = labels
        self.label_ids = torch.tensor(label_ids, dtype=torch.long).to(self.device)
        logging.info(f"Labels: {labels}")
        
    def complete(self, prompts):
        batch = self.tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )
        logging.info(batch[0].ids)
        logging.info(self.tokenizer.decode(batch[0].ids , skip_special_tokens=True) )

        
        if batch["input_ids"].shape[1] > self.tokenizer.max_len_single_sentence:
            prompt_length = batch["input_ids"].shape[1]
            model_max_length = self.tokenizer.max_len_single_sentence

            assert (
                f"prompt length {prompt_length} > "
                f"model_max_length {model_max_length}"
            )
        
        #print("batch = ", batch)
        batch = to_device(batch, self.device)
        input_length = batch["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **batch,
                max_length=input_length + 1,
                output_hidden_states=True,
                output_scores =True,
                do_sample =False,
                return_dict_in_generate =True
            )
        
        encoded = output.sequences
        # print("encoded old== ", encoded.shape)
        # print("encoded[:, input_length:] == ", encoded[:, input_length:].shape)
        decoded = self.tokenizer.batch_decode(encoded[:, input_length:])
        #print("decoded == ", decoded)
        generation_results = []
        logits_all = output.scores[0]
        #print("logits_all == ", logits_all)
        #print("logits_all shape == ", logits_all.shape)
        completion = []
        probs_arr = []
        for i, raw_completion in enumerate(decoded):
            #print("self.label_ids == ", self.label_ids)
            
            logits = logits_all[i, self.label_ids]
            probs= F.softmax(logits, dim=0)
            #print("logits == ", logits)
            pred = logits.argmax(0)
            completion1 = self.labels[pred]
            completion.append(completion1)
            probs_arr.append(probs)
        
        return completion , probs_arr
    
    def complete_all_cali(self , prompts):
        res = [None] * len(prompts)
        probs = [None] * len(prompts)
        
        uncached = []
        for i, prompt in enumerate(prompts):
            uncached.append((i, prompt))
            
        for i in tqdm(range(0, len(uncached), self.batch_size)):
            chunk = uncached[i : i + self.batch_size]
            # print("chunk = ",len(chunk))
            # print(chunk)
            chunk_prompts = [tup[1] for tup in chunk]
            outputs , probs_arr = self.complete(chunk_prompts)
            for (j, prompt), output , probs_array in zip(chunk, outputs , probs_arr):
                res[j] = output.strip()
                probs[j] = probs_array
        return res , probs
    
    def complete_all(self, prompts , prompts_cali , prompts_cali2 , prompts_cali3):
        res = [None] * len(prompts)
        probs = [None] * len(prompts)
        
        uncached = []
        for i, prompt in enumerate(prompts):
            uncached.append((i, prompt))
            
        for i in tqdm(range(0, len(uncached), self.batch_size)):
            chunk = uncached[i : i + self.batch_size]
            # print("chunk = ",len(chunk))
            # print(chunk)
            chunk_prompts = [tup[1] for tup in chunk]
            outputs , probs_arr = self.complete(chunk_prompts)
            for (j, prompt), output , probs_array in zip(chunk, outputs , probs_arr):
                res[j] = output.strip()
                probs[j] = probs_array
            # print("probs_arr == ",probs_arr)
            # print(llllllllllllllllllllllllllllllll)
            
        
        #print("probs == ",probs)
        
        
        
        
        if self.use_calibration:
            res_cali , probs_cali = self.complete_all_cali(prompts_cali)
            res_cali2 , probs_cali2 = self.complete_all_cali(prompts_cali2)
            res_cali3 , probs_cali3 = self.complete_all_cali(prompts_cali3)
            res = [None] * len(prompts)
            
            
            for j , (p_ori , p_cali , p_cali2 ,p_cali3) in enumerate(zip(probs , probs_cali , probs_cali2 , probs_cali3)):
                
                # print(p_cali[0], p_cali2[0],p_cali3[0])
                
                raw_cali_probs = torch.stack([p_cali[0] , p_cali2[0], p_cali3[0]])
                
                #print("raw_cali_probs ==== ",raw_cali_probs)
                W = 1.0 / raw_cali_probs.mean(dim=0)
                #print("W ==== ",W)
                #print("p_ori ====== ",p_ori[0])
                p_new = p_ori[0] * W
                #print("p_new ==== ",p_new)
                p_new = p_new / p_new.sum()
                #print("p_new ==== ",p_new)
                # p_new = p_ori[0] - p_cali[0]
                
                
                pred1 = p_new.argmax(0)
                res[j] = self.labels[pred1].strip()
            
        acc=[]
        for pred,label_test in zip(res,self.label_test):
            print(f"{str(pred)} , {str(label_test)}")
            acc.append(str(pred)==str(label_test))
        
        print(np.mean(acc))
        
        return res
    




MODELS_hf = {
          "llama": "decapoda-research/llama-7b-hf",
          "alpaca": "chavinlo/alpaca-native",
          "alpaca-lora": "chainyo/alpaca-lora-7b",
          }

class LlamaWrapper:
    def initialize_model(cls, model_name):
        return LLaMAForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                )
    def __init__(
        self,
        model_name: str,
        
        batch_size: int = 8,
        k: int = 4,
        use_calibration: bool = False,
        labels: List[str] = None,
        label_test: List[str] = None,
        # calibrate: bool = False,
    ): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device ="cpu"
        if self.device != "cuda":
            logging.warning(f"Cannot find gpu, setting device to cpu.")
        self.batch_size = batch_size
        #self.calibrate = calibrate
        logging.info(f"Setting batch_size={batch_size}")
        model_hf = MODELS_hf[model_name]
        
        self.use_calibration = use_calibration
        self.tokenizer = LLaMATokenizer.from_pretrained(model_hf)
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_bos_token = False
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.label_test = label_test
        
        logging.info(f"Initializing {model_name}")
        self.model_name = model_name
        self.model = self.initialize_model(model_hf)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        for param in self.model.parameters():
            param.requires_grad = False
        #self.model.eval().to(self.device)
        
        label_ids = []
        print("labels == ",labels)
        if labels is not None:
            for label, label_encoded in zip(
                labels,
                self.tokenizer.batch_encode_plus([l for l in labels])[
                    "input_ids"
                ],
            ):
                print(label)
                print(label_encoded)
                label_id = label_encoded[0]
                label_str = self.tokenizer.convert_ids_to_tokens(label_id)
                if len(label_encoded) > 1:
                    logging.warning(
                        f"Cannot find matching id for {label}, using prefix {label_str}"
                    )
                label_ids.append(label_id)

        self.labels = labels
        self.label_ids = torch.tensor(label_ids, dtype=torch.long).to(self.device)
        logging.info(f"Labels: {labels}")
        logging.info(f"label_ids: {label_ids}")
    
    def complete(self, prompts):
        batch = self.tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )

        logging.info(batch[0].ids)
        logging.info(self.tokenizer.decode(batch[0].ids , skip_special_tokens=True) )

        if batch["input_ids"].shape[1] > self.tokenizer.max_len_single_sentence:
            prompt_length = batch["input_ids"].shape[1]
            model_max_length = self.tokenizer.max_len_single_sentence

            assert (
                f"prompt length {prompt_length} > "
                f"model_max_length {model_max_length}"
            )
        generation_config = GenerationConfig(
                            temperature=0.0,
                            top_p=0.95,
                            repetition_penalty=1.15,
                        )
        #print("batch = ", batch)
        batch = to_device(batch, self.device)
        input_length = batch["input_ids"].shape[1]
        
        with torch.no_grad():
            output = self.model.generate(
                **batch,
                max_new_tokens=1,
                output_hidden_states=True,
                generation_config=generation_config,
                output_scores =True,
                do_sample =False,
                return_dict_in_generate =True
            )
            
        encoded = output.sequences
        # print("encoded old== ", encoded.shape)
        # print("encoded[:, input_length:] == ", encoded[:, input_length:].shape)
        decoded = self.tokenizer.batch_decode(encoded[:, input_length:])
        #print("decoded == ", decoded)
        generation_results = []
        logits_all = output.scores[0]
        #print("logits_all == ", logits_all)
        #print("logits_all shape == ", logits_all.shape)
        completion = []
        probs_arr = []
        for i, raw_completion in enumerate(decoded):
            #print("self.label_ids == ", self.label_ids)
            
            logits = logits_all[i, self.label_ids]
            probs= F.softmax(logits, dim=0)
            #print("logits == ", logits)
            pred = logits.argmax(0)
            completion1 = self.labels[pred]
            completion.append(completion1)
            probs_arr.append(probs)
        
        
        return completion , probs_arr
    
    def complete_all_cali(self , prompts):
        res = [None] * len(prompts)
        probs = [None] * len(prompts)
        
        uncached = []
        for i, prompt in enumerate(prompts):
            uncached.append((i, prompt))
            
        for i in tqdm(range(0, len(uncached), self.batch_size)):
            chunk = uncached[i : i + self.batch_size]
            # print("chunk = ",len(chunk))
            # print(chunk)
            chunk_prompts = [tup[1] for tup in chunk]
            outputs , probs_arr = self.complete(chunk_prompts)
            for (j, prompt), output , probs_array in zip(chunk, outputs , probs_arr):
                res[j] = output.strip()
                probs[j] = probs_array
        return res , probs
    
    def complete_all(self, prompts):
        res = [None] * len(prompts)
        probs = [None] * len(prompts)
        uncached = []
        
        for i, prompt in enumerate(prompts):
            uncached.append((i, prompt))
            
        for i in tqdm(range(0, len(uncached), self.batch_size)):
            chunk = uncached[i : i + self.batch_size]
            # print("chunk = ",len(chunk))
            # print(chunk)
            chunk_prompts = [tup[1] for tup in chunk]
            outputs , probs_arr = self.complete(chunk_prompts)
            for (j, prompt), output , probs_arr in zip(chunk, outputs , probs_arr):
                res[j] = output.strip()
                probs[j] = probs_array
        
        if self.use_calibration:
            res_cali , probs_cali = self.complete_all_cali(prompts_cali)
            res_cali2 , probs_cali2 = self.complete_all_cali(prompts_cali2)
            res_cali3 , probs_cali3 = self.complete_all_cali(prompts_cali3)
            res = [None] * len(prompts)
            
            
            for j , (p_ori , p_cali , p_cali2 ,p_cali3) in enumerate(zip(probs , probs_cali , probs_cali2 , probs_cali3)):
                
                # print(p_cali[0], p_cali2[0],p_cali3[0])
                
                raw_cali_probs = torch.stack([p_cali[0] , p_cali2[0], p_cali3[0]])
                
                #print("raw_cali_probs ==== ",raw_cali_probs)
                W = 1.0 / raw_cali_probs.mean(dim=0)
                #print("W ==== ",W)
                #print("p_ori ====== ",p_ori[0])
                p_new = p_ori[0] * W
                #print("p_new ==== ",p_new)
                p_new = p_new / p_new.sum()
                #print("p_new ==== ",p_new)
                # p_new = p_ori[0] - p_cali[0]
                
                
                pred1 = p_new.argmax(0)
                res[j] = self.labels[pred1].strip()
            
        acc=[]
        for pred,label_test in zip(res,self.label_test):
            print(f"{str(pred)} , {str(label_test)}")
            acc.append(str(pred.strip())==str(label_test.strip()))
        
        print(np.mean(acc))
        
        return res
    