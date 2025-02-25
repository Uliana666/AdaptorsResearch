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
accuracy_metric = load('accuracy')


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     print(predictions.shape, labels.shape)
#     print(labels)
#     sm = dict()
#     sm['accuracy'] = 0
#     k = labels.shape[0]
    
#     for i in range(len(predictions)):

#         mask = labels[i] != -100
#         valid_labels = labels[i][mask]
#         valid_predictions = predictions[i][mask]
    
#         valid_predictions = np.roll(valid_predictions, 1)
    

#         accuracy = accuracy_metric.compute(predictions=valid_predictions, references=valid_labels)
#         # print(accuracy)
#         sm['accuracy'] += accuracy['accuracy'] / k
        
#     return sm

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0].reshape(labels.shape)
    # print(predictions.shape, labels.shape)
    # print(labels)
    sm = dict()
    sm['accuracy'] = 0
    k = labels.shape[0]
    
    for i in range(len(predictions)):

        mask = labels[i] != -100
        valid_labels = labels[i][mask]
        valid_predictions = np.roll(predictions[i], 1)[mask]
        
        # print(valid_labels, valid_predictions)
    
        all_correct = all(p == r for p, r in zip(valid_predictions, valid_labels))

        sm['accuracy'] += (1 if all_correct else 0) / k
        
    return sm

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.stack([torch.argmax(logit, dim=-1) for logit in logits])
    return pred_ids, labels


def CalcLoss(dataset, name_text, max_length, model, tokenizer):
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
        start_token=374,
    )

    training_args = TrainingArguments(
        output_dir="./logs/test_new",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=4, 
        fp16=True,
        report_to="tensorboard",
        eval_accumulation_steps=5
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    eval_results = trainer.evaluate()
    return eval_results    
