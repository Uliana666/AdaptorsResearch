import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
from lib import Globals, Models, Datasets
import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
import tqdm
import torch
import torch
from lib import Globals
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator
from evaluate import load


@dataclass
class InferenceArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
        
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
    
    count_examples: int = field(default=None, metadata={"help": "count of examples will be load from dataset"})
        
    start_token: int = field(default=None, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    
    name_dataset: str = field(default="common-reasoning", metadata={"help": "Chose: BoolQ, PIQA, SIQA, hellaswag, winogrande, ARC-E, ARC-C, OBQA"})

    
    
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0].reshape(labels.shape)
    sm = dict()
    sm['accuracy'] = 0
    k = labels.shape[0]
    
    for i in range(len(predictions)):
        mask = labels[i] != -100
        valid_labels = labels[i][mask]
        valid_predictions = np.roll(predictions[i], 1)[mask]
            
        all_correct = all(p == r for p, r in zip(valid_predictions, valid_labels))

        sm['accuracy'] += (1 if all_correct else 0) / k
        
    return sm


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.stack([torch.argmax(logit, dim=-1) for logit in logits])
    return pred_ids, labels


def valid():
    parser = transformers.HfArgumentParser(InferenceArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    model, tokenizer = Models.LoadLLM(script_args.model_name_or_path)

    common_reasoning = Datasets.LoadCommonReasoning('validation', script_args.count_examples, 
                                                    script_args.name_dataset, script_args.seed_of_gen)
    
    dataset = common_reasoning['dataset'].map(
        Globals._tokenize_,
        batched=True,
        remove_columns=common_reasoning['dataset'].column_names,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, 
                "max_length": script_args.model_max_length, 
                "name_text": common_reasoning['name_text']},
        load_from_cache_file=False,
    )
    
    data_collator = Globals.DataCollatorForChat(
        tokenizer=tokenizer,
        mlm=False,
        start_token=script_args.start_token
    )


    trainer = Trainer(
        model=model,
        args=script_args,
        data_collator=data_collator,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == "__main__":
    valid()