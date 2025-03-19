from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_utils import load_sharded_checkpoint


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

def LoadFineTuneLLM(args, device='cuda'):
    model, _ = LoadLLM(args.name)
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = PrepareModel(model, args, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    load_info = load_sharded_checkpoint(
            model=model,
            folder=args.path,
            strict=False
        )
    
    return model, tokenizer

def PrepareModel(model, args, tokenizer, logs=None):
    if is_valid_type_opt(args.mode):
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.rank, 
                            lora_alpha=args.rank * 2, lora_dropout=0, 
                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'],
                            init_lora_weights=args.mode)
        return get_peft_model(model, peft_config)
    
    if args.mode == 'scorda':
        peft_config = SCorDAConfig(r=args.rank, alpha=args.alpha_scorda, dropout=0, init_strategy=args.init_strategy,
                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'],
                            samples=args.samples)
        return get_peft_model_scorda(model, peft_config, args, tokenizer, logs)
        

