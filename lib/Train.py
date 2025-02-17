import numpy as np
import tqdm
import torch
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator


import torch
from transformers import AutoTokenizer




def format_and_tokenize(examples, tokenizer, query, response, prompt, max_length):
    texts = [prompt.format(instruction=q) for q, a in zip(examples[query], examples[response])]
    tks = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # print(tks)
    
    lens = [len(t) for t in tks['input_ids']]
    texts = [prompt.format(instruction=q) + a for q, a in zip(examples[query], examples[response])]
    t =  tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    for i in range(len(lens)):
        t['attention_mask'][i][:lens[i]] = 0
    return t

def Trains(name_dataset, type, count, query, response, prompt, max_length, model, tokenizer):
    dataset = load_dataset(name_dataset, split=type + f"[:{count}]")
    
    dataset = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": query, 
                   "response": response, "prompt": prompt, 
                   "max_length": max_length}
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./logs/test_new",
        do_train=True,
        do_eval=False,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=1, 
        fp16=True,
        report_to="tensorboard",
        eval_accumulation_steps=5,
        num_train_epochs=1,
        logging_steps=10,                       
        logging_dir='./logs/meow',
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    return trainer.train()
