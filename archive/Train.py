import numpy as np
import tqdm
import torch
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm
from lib import Globals
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator


import torch
from transformers import AutoTokenizer



def Trains(dataset, name_text, max_length, model, tokenizer):

    dataset = dataset.map(
        Globals._tokenize_,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "name_text": name_text},
        load_from_cache_file=False,
    )
    
    
    data_collator = Globals.DataCollatorForChat(
        tokenizer=tokenizer,
        mlm=False,
        # start_token=128007
        start_token=374
    )

    training_args = TrainingArguments(
        output_dir="./logs/cats_model",
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=32, 
        fp16=True,
        report_to="tensorboard",
        num_train_epochs=1,
        logging_steps=10,
        learning_rate=2e-4,               
        weight_decay=0.0,                   
        warmup_steps=30,   
        lr_scheduler_type="cosine",
        optim='adamw_torch',               
        logging_dir='./logs/model',
        save_steps=1000,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    return trainer.train()
