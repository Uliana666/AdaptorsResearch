from dataclasses import dataclass, field
from typing import Optional, Literal
from lib import Models
import pickle
import transformers
from omegaconf import OmegaConf


@dataclass
class CalcArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
        
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
            
    start_token: int = field(default=10000, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    
    config_path: str = field(default=None)
    


def calc():
    parser = transformers.HfArgumentParser(CalcArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    model, tokenizer = Models.LoadLLM(script_args.model_name_or_path)

    logs = {}
    adaptor_config = OmegaConf.load(script_args.config_path)
    model = Models.PrepareModel(model, adaptor_config, script_args, tokenizer, False, logs)
        
    print(logs)
    
    with open(script_args.output_dir, 'wb') as file:
        pickle.dump(logs, file)


if __name__ == "__main__":
    calc()