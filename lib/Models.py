from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model



def LoadLLM(name):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map='cuda')

    return model, tokenizer

def LoadLLMLoRA(path):
    tokenizer = AutoTokenizer.from_pretrained(path)

    base_model = AutoModelForCausalLM.from_pretrained(path)

    model = PeftModel.from_pretrained(base_model, path)

    return model, tokenizer

def PrepareLoRA(model, r, alpha, dropout):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r, 
                            lora_alpha=alpha, lora_dropout=dropout, 
                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'])
    return get_peft_model(model, peft_config)
