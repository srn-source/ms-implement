import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

from data import  SST2Processor
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import logging
import copy

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
    
    if  len(ids1)+len(ids2) > max_length:
        print("lennnnnnn = ",len(ids1)+len(ids2))
        
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        # print("len(ids1) = ",len(ids1))
        # print("len(ids2) = ",len(ids2))
        # print("max_length = ",max_length)
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
                
                # print("train_input = ",tokenizer.decode(train_input))
                # print("test_input = ",tokenizer.decode(test_input))
                # print("+++++++++++++++++++++++++++++++++++++++++++++")
                _input_ids, _attention_mask, _token_type_ids = \
                    prepro_sentence_pair_single(train_input, test_input, max_length,
                                                bos_token_id, eos_token_id,
                                                allow_truncation=allow_truncation)
                input_ids.append(_input_ids)
                attention_mask.append(_attention_mask)
                token_type_ids.append(_token_type_ids)
    else:
        for input_ch , label_ch in zip(train_inputs, test_inputs):
            # print("train_input = ",tokenizer.decode(input_ch))
            # print("test_input = ",tokenizer.decode(label_ch))
            # print("+++++++++++++++++++++++++++++++++++++++++++++")
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

def get_dataloader(inputs, batch_size , k):

    shape = inputs["input_ids"].shape
    for v in inputs.values():
        assert v.shape==shape

    dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"])
    if k > 0:
        sampler=SequentialSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def inference(model, inputs, batch_size,k,return_logits=False):
    #print("inputs inference= ",len(inputs))
    dataloader = get_dataloader(inputs, batch_size,k)
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

def flatten_label_losses(label_losses, dev_data):
    print("label_losses[0] ========= ",len(label_losses[0]))
    print("label_losses[1] ========= ",len(label_losses[1]))
    print("len label_losses" , len(label_losses))
    print("len(dev_data) = ",len(dev_data))
    # print("label_losses ========= ",label_losses[0])
    for label in range(len(label_losses)):
        k = int(len(label_losses[label]) / len(dev_data))
        print("k = ",k)
        print("label = ",label)
        label_losses[label] = [label_losses[label][k*i:k*(i+1)] for i in range(len(dev_data))]
        
        print("label_losses ==== test1 ===",label_losses[label][k*0:k*(0+1)])
    print("label_losses [0] ========= ",len(label_losses[0]))
    print("label_losses [1] ========= ",len(label_losses[1]))
    print("label_losses [0][0] ========= ",label_losses[0][0])
    return label_losses

@torch.no_grad()
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
        # print("label = ", label)
        # print("prediction = ",prediction)
        
        # print("label = ", type(label))
        # print("prediction = ",type(prediction))
        
        # if prediction == str(label):
        #     print("prediction == label => ", label)
        acc.append(str(prediction)==str(label))
    #print("acc orig = ", acc)
    return np.mean(acc)

def main(args):
    
    seed_every_thing(args.train_seed)
    
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    # if args.model_name =="t5-small":
    #     model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    #     tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    # else:
    #     model = GPT2LMHeadModel.from_pretrained(args.model_name)
    #     tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    
    for param in model.parameters():
        param.requires_grad = False
                    
    max_length = 256
    batch_size = args.batch_size
    
    

    
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    main_class = SST2Processor(args.k, args.train_seed, args.dataset ,tokenizer, args.kate_metric , args.reversed , args.encoder_kate , args.use_calibration , args.ensemble )
    test_inputs_with_label , test_inputs_token , data_token_with_space = main_class.generate_test_set()
    # dev_data = []
    
    # train_data_for_fewshot = SST2Processor(args.k, args.seed, args.dataset , tmp,tmp_idx, tokenizer , "Train")
    # demonstrations , test_inputs, dev_data = train_data_for_fewshot.generate_datasets()
    group_id_kate = []
    if args.kate:
        group_id_kate = main_class.kate_process()
        
    #print("group_id_kate === ", group_id_kate)
    
    accs = []
    for idx, (tmp,tmp_idx) in enumerate(zip(templates,templates_idx)):
       
        #print(test_inputs[0:5])
        #print("hi")
        input_tensors = []
        input_tensors_calibration = []
        #train_data_for_fewshot = SST2Processor(args.k, args.seed, args.dataset , tmp,tmp_idx, tokenizer , "Train")
        label_ensemble = []
        
        for i in label_words:
            if args.method == "direct":
                
                prefix = tokenizer(" " +(tmp % i))["input_ids"] 
                na_token = tokenizer("N/A")["input_ids"]
                
                if args.kate : #not support zero shot
                    assert args.k > 0
                    if args.ensemble:
                        prompt , prompt_calibration= main_class.ensemble_generate_promt(test_inputs_token , prefix[:tmp_idx] , tmp , group_id_kate)
                    else:
                        prompt , prompt_calibration= main_class.kate_generate_promt(group_id_kate , test_inputs_token , prefix[:tmp_idx] , tmp)
                        print("prompt = ",len(prompt))
                        
                else:
                    if args.k > 0:
                        if args.ensemble:
                            prompt , prompt_calibration= main_class.ensemble_generate_promt(test_inputs_token , prefix[:tmp_idx] , tmp )
                        else:
                            demonstrations  = main_class.generate_setOfDemon(tmp)
                            prompt = [demonstrations.copy() + test_input + prefix[:tmp_idx] for test_input in test_inputs_token]
                            prompt_calibration = [demonstrations.copy() + na_token + prefix[:tmp_idx] for test_input in test_inputs_token]
                    else: #zero shot case
                        assert not args.ensemble
                        prompt = [test_input + prefix[:tmp_idx] for test_input in test_inputs_token]
                        prompt_calibration = [na_token + prefix[:tmp_idx] for test_input in test_inputs_token]
                    
                    print("prompt = ",len(prompt))
                    
                tensor = prepro_sentence_pair(prompt,
                                                [prefix[tmp_idx:-1]], max_length,
                                                bos_token_id, eos_token_id,  method = "direct",
                                                allow_truncation=True )
                input_tensors.append(tensor)
                
                if args.use_calibration:
                     
                    tensor_calibrate = prepro_sentence_pair(prompt_calibration,
                                                [prefix[tmp_idx:-1]], max_length,
                                                bos_token_id, eos_token_id, method = "direct",
                                                allow_truncation=True)
                    input_tensors_calibration.append(tensor_calibrate)
                
                
            elif args.method == "channel":
                assert not args.use_calibration
                prefix = tokenizer(" "+(tmp % i))["input_ids"] 
                #na_token = tokenizer("N/A")["input_ids"]
                label_ = []
                if args.kate :
                    assert args.k > 0
                    if args.ensemble:
                        prompt, label_ = main_class.kate_generate_promt_channel( data_token_with_space , prefix , tmp, group_id_kate)
                        print("prompt = ",len(prompt))
                    else:
                        #demonstrations  = main_class.generate_setOfDemon_channel(tmp, group_id_kate)
                        #prompt = [demonstrations.copy() + prefix for test_input in data_token_with_space]
                        prompt, label_ = main_class.kate_generate_promt_channel( data_token_with_space , prefix , tmp, group_id_kate)
                        print("prompt = ",len(prompt))
                else:
                    if args.k > 0:
                        if args.ensemble:
                            prompt, label_ = main_class.kate_generate_promt_channel( data_token_with_space , prefix , tmp)
                        else:
                            demonstrations  = main_class.generate_setOfDemon_channel(tmp)
                            prompt = [demonstrations.copy() + prefix for test_input in data_token_with_space]
                    else: #zero shot case
                        assert not args.ensemble
                        prefix = tokenizer((tmp % i))["input_ids"] 
                        prompt = [prefix for test_input in data_token_with_space]
                
                label1 = label_ if len(label_)!= 0 else data_token_with_space
                assert len(prompt) == len(label1)
                tensor = prepro_sentence_pair(prompt ,label1, max_length,
                                            bos_token_id, eos_token_id, method = "channel",
                                            allow_truncation=True)
                input_tensors.append(tensor)
                
                # if args.use_calibration:
                #     tensor_calibrate = prepro_sentence_pair(prompt , [na_token] , max_length,
                #                             bos_token_id, eos_token_id, method = "channel",
                #                             allow_truncation=True)
                #     input_tensors_calibration.append(tensor_calibrate)
        
        for i in range(args.k +1):
            logging.info("Checking the first example...")
            input_ids = input_tensors[0]["input_ids"][i].numpy().tolist()
            token_type_ids = input_tensors[0]["token_type_ids"][i].numpy().tolist()
            logging.info("Input:")
            logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            logging.info("Output:")
            logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
            
        logging.info("=========================================================")
        for i in range(args.k +1):
            
            logging.info("Checking the Second example...")
            input_ids = input_tensors[1]["input_ids"][i].numpy().tolist()
            token_type_ids = input_tensors[1]["token_type_ids"][i].numpy().tolist()
            logging.info("Input:")
            logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            logging.info("Output:")
            logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
            
            # logging.info("=================================================================================================")
            # logging.info("Checking the first example...")
            # input_ids = input_tensors[0]["input_ids"][1].numpy().tolist()
            # token_type_ids = input_tensors[0]["token_type_ids"][1].numpy().tolist()
            # logging.info("Input:")
            # logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            # logging.info("Output:")
            # logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
            # logging.info("==================================================")
            # logging.info("Checking the Second example...")
            # input_ids = input_tensors[1]["input_ids"][1].numpy().tolist()
            # token_type_ids = input_tensors[1]["token_type_ids"][1].numpy().tolist()
            # logging.info("Input:")
            # logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            # logging.info("Output:")
            # logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
        #print("input_tensors = ",len(input_tensors))

        if args.use_calibration:
            logging.info("Checking the first example... use_calibration")
            input_ids = input_tensors_calibration[0]["input_ids"][0].numpy().tolist()
            token_type_ids = input_tensors_calibration[0]["token_type_ids"][0].numpy().tolist()
            logging.info("Input:")
            logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            logging.info("Output:")
            logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
            logging.info("==================================================")
            logging.info("Checking the Second example...")
            input_ids = input_tensors_calibration[1]["input_ids"][0].numpy().tolist()
            token_type_ids = input_tensors_calibration[1]["token_type_ids"][0].numpy().tolist()
            logging.info("Input:")
            logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            logging.info("Output:")
            logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
            logging.info("=================================================================================================")
            logging.info("Checking the first example...")
            input_ids = input_tensors_calibration[0]["input_ids"][1].numpy().tolist()
            token_type_ids = input_tensors_calibration[0]["token_type_ids"][1].numpy().tolist()
            logging.info("Input:")
            logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            logging.info("Output:")
            logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
            logging.info("==================================================")
            logging.info("Checking the Second example...")
            input_ids = input_tensors_calibration[1]["input_ids"][1].numpy().tolist()
            token_type_ids = input_tensors_calibration[1]["token_type_ids"][1].numpy().tolist()
            logging.info("Input:")
            logging.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
            logging.info("Output:")
            logging.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
        
        losses = []
       
        for input_tensor in input_tensors:
                loss_infer = inference(model,
                                        input_tensor,
                                        batch_size , args.k)
                losses.append(loss_infer)
        
        if args.ensemble:
            losses = flatten_label_losses(losses, test_inputs_token)
            
        if args.use_calibration:
            losses_calibration = []
            for input_tensor in input_tensors_calibration:
                loss_infer = inference(model,
                                        input_tensor,
                                        batch_size , args.k)
                losses_calibration.append(loss_infer)
            
            if args.ensemble:
                losses_calibration = flatten_label_losses(losses_calibration, test_inputs_token)
            
            losses1 = copy.deepcopy(losses)
            for i, (bias_loss, loss) in enumerate(zip(losses_calibration, losses1)):
                loss = np.array(loss)
                bias_loss = np.array(bias_loss)
                losses[i] = loss - bias_loss
            
        #print("loss len= ", len(losses))
        acc = evaluate(test_inputs_with_label, {str(i): loss for i, loss in enumerate(losses)})
        accs.append(acc)
        logging.info(f"Acc = {acc}")
        
    logging.info("Accuracy = %.1f (Avg) / %.1f (Worst)" % (100*np.mean(accs), 100*np.min(accs)))
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes" , help="SetFit/sst2, rotten_tomatoes")
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--model_name", type=str, default="gpt2-large")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--train_seed", type=int, default=87 , help="{13|21|42|87|100}")
    parser.add_argument("--batch_size", type=int, default=12 )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--kate", action='store_true', help='enable kate' )
    parser.add_argument("--kate_metric", type=str, default="cosine"  ,help="euclidean or cosine" )
    parser.add_argument('--encoder_kate', default='roberta-base', type=str, help='roberta-base, roberta-large')
    parser.add_argument("--reversed", action='store_true', help='cosine kate reversed' )
    parser.add_argument("--use_calibration", default=False, action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)