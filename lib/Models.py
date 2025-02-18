from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def LoadLLM(name):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map='cuda')
    
    special_token = None
    if name == "meta-llama/Llama-3.2-1B-Instruct":
        special_token = tokenizer.convert_tokens_to_ids('<|end_header_id|>')

    return {"model" : model, "tokenizer": tokenizer, "special_token": special_token}