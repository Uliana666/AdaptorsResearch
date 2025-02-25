import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
from archive import Train
from lib import Globals, Models, Datasets
import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
        
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
    
    count_examples: int = field(default=10000, metadata={"help": "count of examples will be load from dataset"})
    
    optim: str = field(default="adamw_torch")
    
    start_token: int = field(default=10000, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    rank: int = field(default=None, metadata={"help": "The rank of adapter."})
    mode: Literal["lora", "pissa", "corda", "my"] = field(
        default="lora", metadata={"help": "Use type of adaptor: lora, pissa, corda, my"}
    )

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    model, tokenizer = Models.LoadLLM(script_args.model_name_or_path)

    common_reasoning = Datasets.LoadCommonReasoning('train', script_args.count_examples, seed=script_args.seed_of_gen)
    
    if script_args.mode == "lora":
        print("You chose lora")
        model = Models.PrepareLoRA(model, script_args.rank, script_args.rank * 2, 0)
    elif script_args.mode == "pissa":
        print("You chose pissa")
        model = Models.PreparePiSSA(model, script_args.rank, script_args.rank * 2, 0)
        
    dataset = common_reasoning['dataset'].map(
        Globals._tokenize_,
        batched=True,
        remove_columns=common_reasoning['dataset'].column_names,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, 
                "max_length": script_args.model_max_length, 
                "name_text": common_reasoning['name_text']},
        load_from_cache_file=False,
    )
    
    
    data_collator = Globals.DataCollatorForChat(
        tokenizer=tokenizer,
        mlm=False,
        start_token=script_args.start_token
    )


    trainer = Trainer(
        model=model,
        args=script_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir,'ft'))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir,'ft'))



if __name__ == "__main__":
    train()