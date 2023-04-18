import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import copy
from typing import Dict, List, Optional
# from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from tqdm import tqdm

logging.basicConfig(level = logging.INFO)

def to_device(tensor_dict, device):
    return {k: v.to(device) for k, v in tensor_dict.items()}

class GPT2Wrapper:
    def initialize_model(cls, model_name):
        return AutoModelForCausalLM.from_pretrained(model_name)
    def __init__(
        self,
        model_name: str,
        
        batch_size: int = 8,
        k: int = 4,
        labels: List[str] = None,
        label_test: List[str] = None,
        # calibrate: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logging.warning(f"Cannot find gpu, setting device to cpu.")
        self.batch_size = batch_size
        #self.calibrate = calibrate
        logging.info(f"Setting batch_size={batch_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_test = label_test
        
        logging.info(f"Initializing {model_name}")
        self.model_name = model_name
        self.model = self.initialize_model(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        for param in self.model.parameters():
            param.requires_grad = False
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
        for i, raw_completion in enumerate(decoded):
            #print("self.label_ids == ", self.label_ids)
            
            logits = logits_all[i, self.label_ids]
            
            #print("logits == ", logits)
            pred = logits.argmax(0)
            completion1 = self.labels[pred]
            completion.append(completion1)
        
        
        return completion
    def complete_all(self, prompts):
        res = [None] * len(prompts)
        uncached = []
        for i, prompt in enumerate(prompts):
            uncached.append((i, prompt))
            
        for i in range(0, len(uncached), self.batch_size):
            chunk = uncached[i : i + self.batch_size]
            # print("chunk = ",len(chunk))
            # print(chunk)
            chunk_prompts = [tup[1] for tup in chunk]
            outputs = self.complete(chunk_prompts)
            for (j, prompt), output in zip(chunk, outputs):
                res[j] = output.strip()

            
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
                                                device_map="auto",
                                                )
    def __init__(
        self,
        model_name: str,
        
        batch_size: int = 8,
        k: int = 4,
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
        for i, raw_completion in enumerate(decoded):
            #print("self.label_ids == ", self.label_ids)
            
            logits = logits_all[i, self.label_ids]
            
            #print("logits == ", logits)
            pred = logits.argmax(0)
            completion1 = self.labels[pred]
            completion.append(completion1)
        
        
        return completion
    def complete_all(self, prompts):
        res = [None] * len(prompts)
        uncached = []
        for i, prompt in enumerate(prompts):
            uncached.append((i, prompt))
            
        for i in tqdm(range(0, len(uncached), self.batch_size)):
            chunk = uncached[i : i + self.batch_size]
            # print("chunk = ",len(chunk))
            # print(chunk)
            chunk_prompts = [tup[1] for tup in chunk]
            outputs = self.complete(chunk_prompts)
            for (j, prompt), output in zip(chunk, outputs):
                res[j] = output.strip()

            
        acc=[]
        for pred,label_test in zip(res,self.label_test):
            print(f"{str(pred)} , {str(label_test)}")
            acc.append(str(pred.strip())==str(label_test.strip()))
        
        print(np.mean(acc))
        
        return res