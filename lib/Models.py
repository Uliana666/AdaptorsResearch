from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
from omegaconf import OmegaConf
from dataclasses import dataclass, fields


from super_corda.Config import SCorDAConfig
from super_corda.Config import get_peft_model as get_peft_model_scorda

def LoadLLM(name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device)

    return model, tokenizer


def LoadFineTuneLLM(config, args):
    model, _ = LoadLLM(args.name)
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = PrepareModel(model, config, args, tokenizer, True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    _ = load_sharded_checkpoint(
            model=model,
            folder=args.path,
            strict=False
        )
    
    return model, tokenizer

def PrepareModel(model, config_base, args, tokenizer, only_init = False, logs=None):
    
    config_dict = {f.name: config_base.get(f.name) for f in fields(SCorDAConfig)}
    config = SCorDAConfig(**config_dict)
    
    if only_init:
        config.init_strategy = "lora"

    return get_peft_model_scorda(model, config, args, tokenizer, logs)
    

