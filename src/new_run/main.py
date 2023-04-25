import argparse
# from data import  SST2Processor
# from model import GPT2Wrapper
import numpy as np
import random
import torch
import logging
import copy
# from new_run import MODELS, PROCESSORS
from data import (
    BaseProcessor,
    SST2Processor,
    AGNewsProcessor
)

from model import (
    GPT2Wrapper,
    LlamaWrapper,
    GPT3Wrapper
)

PROCESSORS = {
    "sst2": SST2Processor,
    "agnews": AGNewsProcessor,
    
}


MODELS = {"gpt2": GPT2Wrapper,
          "gpt2-medium": GPT2Wrapper,
          "gpt2-large": GPT2Wrapper,
          "gpt2-xl": GPT2Wrapper,
          "stablelm": GPT2Wrapper,
          "llama": LlamaWrapper,
          "alpaca": LlamaWrapper,
          "alpaca-lora": LlamaWrapper,
          "gpt_j6b": GPT2Wrapper,
          "gpt4all_j": GPT2Wrapper,
          #"gpt4all_lora": GPT2Wrapper,
          "dolly_v2_7b": GPT2Wrapper,
         "gpt3":GPT3Wrapper
          }

logging.basicConfig(level = logging.INFO)


def seed_every_thing(train_seed):
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(train_seed)

def main(args):
    seed_every_thing(args.train_seed)
    
    
    #model_send = MODELS[args.model_name](args.model_name , args.batch_size, args.k , args.kate, args.use_calibration, **processor.model_kwargs)
    processor = PROCESSORS[args.dataset](args.train_seed, args.k , args.kate, args.kate_metric , args.reversed , args.model_name, args.entropy_ordering)
    prompts, prompts_cali , prompts_cali2 , prompts_cali3 = processor.create_prompt(args.model_name)
    model_type = MODELS[args.model_name](args.model_name , args.batch_size, args.k , args.kate, args.use_calibration, **processor.model_kwargs)
    out_res = model_type.complete_all(prompts , prompts_cali,prompts_cali2 , prompts_cali3)
    #print(out_res)
    # if "gpt2" in args.model_name:
    #     init_model = GPT2Wrapper(args.model_name , args.batch_size, args.k)
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2" , help="SetFit/sst2, agnews")
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--model_name", type=str, default="gpt3" , help="{gpt2|gpt2-medium|gpt2-large|llama|alpaca|alpaca-lora}")
    parser.add_argument("--entropy_ordering", default=False, action="store_true")
    parser.add_argument("--train_seed", type=int, default=87 , help="{13|21|42|87|100}")
    parser.add_argument("--batch_size", type=int, default=16 )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--kate", action='store_true', help='enable kate' )
    parser.add_argument("--kate_metric", type=str, default="euclidean"  ,help="euclidean or cosine" )
    parser.add_argument('--encoder_kate', default='roberta-base', type=str, help='roberta-base, roberta-large')
    parser.add_argument("--reversed", action='store_true', help='cosine kate reversed' )
    parser.add_argument("--use_calibration", default=False, action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)