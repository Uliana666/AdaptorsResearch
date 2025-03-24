from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
from omegaconf import OmegaConf
from dataclasses import dataclass, fields


from cotan.Config import CoTAnConfig
from cotan.Config import get_peft_model

def LoadLLM(name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device)

    return model, tokenizer


def LoadFineTuneLLM(config, name, path):
    model, _ = LoadLLM(name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = PrepareModel(model, config, None, tokenizer, True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    _ = load_sharded_checkpoint(
            model=model,
            folder=path,
            strict=False
        )
    
    return model, tokenizer

def PrepareModel(model, config_base, args, tokenizer, only_init = False, logs=None):
    
    config_dict = {f.name: config_base.get(f.name) for f in fields(CoTAnConfig)}
    config = CoTAnConfig(**config_dict)
    
    if only_init:
        config.init_strategy = "lora"

    return get_peft_model(model, config, args, tokenizer, logs)
    

