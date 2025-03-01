from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model

from super_corda.Config import SCorDAConfig
from super_corda.Config import get_peft_model as get_peft_model_scorda

def is_valid_type_opt(type_opt):
    valid_types = ["gaussian", "eva", "olora", "pissa", "loftq"]
    if type_opt in valid_types:
        return True
    
    if type_opt.startswith("pissa_niter_"):
        number_part = type_opt[len("pissa_niter_"):]
        if number_part.isdigit():
            return True
    
    return False

def LoadLLM(name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device)

    return model, tokenizer

def PrepareModel(model, r, alpha, dropout, type_opt, special_params=None):
    if is_valid_type_opt(type_opt):
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r, 
                            lora_alpha=alpha, lora_dropout=dropout, 
                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'],
                            init_lora_weights=type_opt)
        return get_peft_model(model, peft_config)
    
    if type_opt == 'scorda':
        peft_config = SCorDAConfig(r=r, alpha=alpha, dropout=dropout, init_strategy='lora',
                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'])
        return get_peft_model_scorda(model, peft_config)
        

