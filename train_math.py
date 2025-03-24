import copy
import os
from dataclasses import dataclass, field
from typing import Optional
from lib import Models, Datasets
import torch
import transformers
from omegaconf import OmegaConf
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl
from torch.utils.tensorboard import SummaryWriter



class LogWeightsCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        print('CALLLL')
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.writer.add_scalar(f'grad_norms/{name}', torch.norm(param.grad), state.global_step)
                # self.writer.add_histogram(f'weights/{name}', param, state.global_step)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    
    logging_path: str = field(default=None)
        
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
    
    count_examples: int = field(default=10000, metadata={"help": "count of examples will be load from dataset"})
    
    optim: str = field(default="adamw_torch")
    
    start_token: int = field(default=10000, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    
    config_path: str = field(default=None)

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    model, tokenizer = Models.LoadLLM(script_args.model_name_or_path)

    common_reasoning = Datasets.LoadCommonReasoning('train', script_args.count_examples, name="MATH", seed=script_args.seed_of_gen)
    
    adaptor_config = OmegaConf.load(script_args.config_path)
    
    model = Models.PrepareModel(model, adaptor_config, script_args, tokenizer)
        
    dataset, data_collator = Datasets.PrepareDataset(**common_reasoning, args=script_args, 
                                                    tokenizer=tokenizer, desc="train")

    if script_args.logging_path != None:
        writer = SummaryWriter(script_args.logging_path)


        trainer = Trainer(
            model=model,
            args=script_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=tokenizer,
            callbacks=[LogWeightsCallback(writer)]
        )
    
        trainer.train()
        trainer.save_state()
        model.save_pretrained(os.path.join(script_args.output_dir,'ft'), max_shard_size='2GB')
        tokenizer.save_pretrained(os.path.join(script_args.output_dir,'ft'))

        writer.close()
    else:
        
        trainer = Trainer(
            model=model,
            args=script_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
    
        trainer.train()
        trainer.save_state()
        model.save_pretrained(os.path.join(script_args.output_dir,'ft'), max_shard_size='2GB')
        tokenizer.save_pretrained(os.path.join(script_args.output_dir,'ft'))



if __name__ == "__main__":
    train()