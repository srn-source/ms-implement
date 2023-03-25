import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data import  SST2Processor
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import logging


logging.basicConfig(level = logging.INFO)

label_words = ["terrible", "great"]
templates = ["A %s one . ", "It was %s . ",
                     "All in all %s . ", "A %s piece . "]
templates_idx = [1,2,3,1]
def seed_every_thing(train_seed):
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(train_seed)

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id, 
                                allow_truncation=True):

    if bos_token_id is not None:
        ids1 = [bos_token_id] + ids1
    if eos_token_id is not None:
        ids2 = ids2 + [eos_token_id] #remove last space
        
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length


    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    
    return input_ids, attention_mask, token_type_ids


def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id, method = "direct",
                         allow_truncation=True):
    input_ids, attention_mask, token_type_ids = [], [], []
    
    if method == "direct":
        for test_input in test_inputs:
            for train_input in train_inputs:
                
                _input_ids, _attention_mask, _token_type_ids = \
                    prepro_sentence_pair_single(train_input, test_input, max_length,
                                                bos_token_id, eos_token_id,
                                                allow_truncation=allow_truncation)
                input_ids.append(_input_ids)
                attention_mask.append(_attention_mask)
                token_type_ids.append(_token_type_ids)
    else:
        for input_ch , label_ch in zip(train_inputs, test_inputs):
            
            _input_ids, _attention_mask, _token_type_ids = \
                    prepro_sentence_pair_single(input_ch, label_ch, max_length,
                                                bos_token_id, eos_token_id,
                                                allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)
                
            
        
    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

def get_dataloader(inputs, batch_size):

    shape = inputs["input_ids"].shape
    for v in inputs.values():
        assert v.shape==shape

    dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"])

    sampler=SequentialSampler(dataset)
        #sampler=RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def inference(model, inputs, batch_size,return_logits=False):
    #print("inputs inference= ",len(inputs))
    dataloader = get_dataloader(inputs, batch_size)
    #print("dataloader = " , dataloader)
    all_losses = []
    for batch in tqdm(dataloader):
        input_ids=batch[0].cuda()
        attention_mask=batch[1].cuda()
        token_type_ids=batch[2].cuda()
        #print("len(batch) = ", len(batch))
        labels=None
        model = model.cuda()
        with torch.no_grad():
            loss = run_model(model, input_ids, attention_mask, token_type_ids,
                             labels=labels, return_logits=return_logits)
            #print("loss inference =",loss)
        all_losses += loss.cpu().detach().numpy().tolist()

    return all_losses

def run_model(model, input_ids, attention_mask, token_type_ids,
              labels=None, return_logits=False):
    #print("llll")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()
    #print("logits ===> ",logits.shape)

    if labels is None:
        labels = input_ids
    labels = labels[..., 1:].contiguous()
    label_mask = token_type_ids[..., 1:].contiguous()
    
    # print("labels ===> ",labels.shape)
    # print("label_mask ===> ",label_mask.shape)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    losses = loss_fct(logits.view(-1, logits.size(-1)),
                      labels.view(-1)) # [batch_size, length]
    losses = losses.view(logits.size(0), logits.size(1)) * label_mask
    
    
    # new_en = test_remove(model,tokenizer)
    # print("new_en ===> ",new_en)
    
    return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)


def evaluate(dev_data, label_losses):
    labels = list(label_losses.keys())
    #print("labels = ", labels) #[0,1]
    acc = []
    
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l:np.sum(label_losses[l][idx]) for l in label_losses}
        #print("label_loss = ",label_loss)
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        #print("label = ", label)
        #print("prediction = ",prediction)
        
        # print("label = ", type(label))
        # print("prediction = ",type(prediction))
        
        # if prediction == str(label):
        #     print("prediction == label => ", label)
        acc.append(str(prediction)==str(label))
    #print("acc orig = ", acc)
    return np.mean(acc)

def main(args):
    
    seed_every_thing(args.train_seed)
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    model = GPT2LMHeadModel.from_pretrained(args.gpt2)
    
    for param in model.parameters():
        param.requires_grad = False
                    
    max_length = 600
    batch_size = args.batch_size
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    main_class = SST2Processor(args.k, args.seed, args.dataset ,tokenizer )
    test_inputs_with_label , test_inputs_token , data_token_with_space = main_class.generate_test_set()
    # dev_data = []
    
    # train_data_for_fewshot = SST2Processor(args.k, args.seed, args.dataset , tmp,tmp_idx, tokenizer , "Train")
    # demonstrations , test_inputs, dev_data = train_data_for_fewshot.generate_datasets()
    
    
    
    accs = []
    for idx, (tmp,tmp_idx) in enumerate(zip(templates,templates_idx)):
       
        #print(test_inputs[0:5])
        #print("hi")
        input_tensors = []
        
        #train_data_for_fewshot = SST2Processor(args.k, args.seed, args.dataset , tmp,tmp_idx, tokenizer , "Train")
        
        
        for i in label_words:
            if args.method == "direct":
                
                demonstrations  = main_class.generate_setOfDemon(tmp)
                
                prefix = tokenizer(" " +(tmp % i))["input_ids"] 
                prompt = [demonstrations.copy() + test_input + prefix[:tmp_idx] for test_input in test_inputs_token]
                print("prompt = ",len(prompt))
                tensor = prepro_sentence_pair(prompt,
                                                [prefix[tmp_idx:-1]], max_length,
                                                bos_token_id, eos_token_id, method = "direct",
                                                allow_truncation=True)
                input_tensors.append(tensor)
                
            elif args.method == "channel":
                demonstrations  = main_class.generate_setOfDemon_channel(tmp)
                
                prefix = tokenizer(" "+(tmp % i))["input_ids"] 
                
                # demon_direct = []
                # out_direct = []
                # for u in data_token_with_space:
                #     demon_direct.append(demonstrations.copy() + prefix)
                #     out_direct.append(u)
                
                # logging.info("channel INPUT:")
                # logging.info(tokenizer.decode(demon_direct[0]))
                # logging.info(tokenizer.decode(out_direct[0]))
                prompt = [demonstrations.copy() + prefix for test_input in data_token_with_space]
                
                # print("demon_direct = ",len(demon_direct))
                # print("out_direct = ",len(out_direct))
                
                tensor = prepro_sentence_pair(prompt , data_token_with_space , max_length,
                                            bos_token_id, eos_token_id, method = "channel",
                                            allow_truncation=True)
                input_tensors.append(tensor)
                
        logging.info("Checking the first example...")
        input_ids = input_tensors[0]["input_ids"][0].numpy().tolist()
        token_type_ids = input_tensors[0]["token_type_ids"][0].numpy().tolist()
        logging.info("Input:")
        logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
        logging.info("Output:")
        logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
        logging.info("==================================================")
        logging.info("Checking the Second example...")
        input_ids = input_tensors[1]["input_ids"][0].numpy().tolist()
        token_type_ids = input_tensors[1]["token_type_ids"][0].numpy().tolist()
        logging.info("Input:")
        logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
        logging.info("Output:")
        logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
        
        #print("input_tensors = ",len(input_tensors))
        losses = []
        for input_tensor in input_tensors:
                loss_infer = inference(model,
                                        input_tensor,
                                        batch_size)
                losses.append(loss_infer)
        
        acc = evaluate(test_inputs_with_label, {str(i): loss for i, loss in enumerate(losses)})
        accs.append(acc)
        logging.info(f"Acc = {acc}")
        
    logging.info("Accuracy = %.1f (Avg) / %.1f (Worst)" % (100*np.mean(accs), 100*np.min(accs)))
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SST2")
    parser.add_argument("--method", type=str, default="channel")
    parser.add_argument("--gpt2", type=str, default="gpt2-large")
    parser.add_argument("--seed", type=str, default="48")
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k", type=int, default=16)
    
    args = parser.parse_args()
    print(args)
    main(args)