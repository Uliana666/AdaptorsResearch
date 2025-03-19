import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
from archive import Train
from lib import Globals, Models, Datasets
import torch
import pickle
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

@dataclass
class CalcArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
        
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
            
    start_token: int = field(default=10000, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    rank: int = field(default=None, metadata={"help": "The rank of adapter."})
    mode: Literal["lora", "pissa", "scorda"] = field(
        default="lora", metadata={"help": "Use type of adaptor: lora, pissa, scorda"}
    )
    
    init_strategy: str = field(default="lora")
    samples: str = field(default=None)
    
    alpha_scorda: int = field(default=None)


def calc():
    parser = transformers.HfArgumentParser(CalcArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    model, tokenizer = Models.LoadLLM(script_args.model_name_or_path)

    logs = {}
    
    model = Models.PrepareModel(model, script_args, tokenizer, logs)
        
    print(logs)
    
    with open(script_args.output_dir, 'wb') as file:
        pickle.dump(logs, file)



if __name__ == "__main__":
    calc()