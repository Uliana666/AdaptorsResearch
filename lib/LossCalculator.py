import numpy as np
import tqdm
import torch
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator



from evaluate import load
accuracy_metric = load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(predictions.shape, labels.shape)
    sm = dict()
    sm['accuracy'] = 0
    k = labels.shape[0]
    
    for i in range(len(predictions)):

        mask = labels[i] != -100
        valid_labels = labels[i][mask]
        valid_predictions = predictions[i][mask]
    
        valid_predictions = np.roll(valid_predictions, 1)
    

        accuracy = accuracy_metric.compute(predictions=valid_predictions, references=valid_labels)
        # print(accuracy)
        sm['accuracy'] += accuracy['accuracy'] / k
        
    return sm

# def format_and_tokenize(examples, tokenizer, query, response, prompt, max_length):
#     texts = [prompt.format(instruction=q) for q, a in zip(examples[query], examples[response])]
#     tks = tokenizer(
#         texts,
#         truncation=True,
#         max_length=max_length,
#         padding="max_length",
#         return_tensors="pt"
#     )
    
#     # print(tks)
    
#     lens = [len(t) for t in tks['input_ids']]
#     texts = [prompt.format(instruction=q) + a for q, a in zip(examples[query], examples[response])]
#     t =  tokenizer(
#         texts,
#         truncation=True,
#         max_length=max_length,
#         padding="max_length",
#         return_tensors="pt"
#     )
#     for i in range(len(lens)):
#         t['attention_mask'][i][:lens[i]] = 0
#     return t

def format_and_tokenize_eval(examples, tokenizer, query, response, prompt, max_length):
    instructions = [
        ({'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt + examples[query][i]},
        {'role': 'assistant', 'content': examples[response][i]}) for i in range(len(examples))
    ]

    texts = tokenizer.apply_chat_template(
        instructions,
        tokenize=False
    )
    
    # print(texts)
    
    t =  tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return t

def CalcLoss(name_dataset, type, count, query, response, prompt, max_length, model, tokenizer):
    dataset = load_dataset(name_dataset, split=type + f"[:{count}]")
    # dataset = load_dataset('Idavidrein/gpqa', 'gpqa_extended', split='train[:10]')
    
    dataset = dataset.map(
        format_and_tokenize_eval,
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
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=1, 
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
    )
    eval_results = trainer.evaluate()
    return eval_results
