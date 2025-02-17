from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def LoadLLM(name):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map='cuda')

    return model, tokenizer