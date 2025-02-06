import numpy as np
import tqdm
import torch
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm


class TextExampleDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def clc_loss_example(model, tokenizer, example):
    device = next(model.parameters()).device
    inputs = tokenizer(example, return_tensors='pt', truncation=True).to(device)

    with torch.no_grad():
        # outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'].clone(), attention_mask=inputs['attention_mask'])
        loss = outputs.loss

    return loss.item()



def collate_fn(batch):
    return {"text": [item['text'] for item in batch]}


# def clc_loss_batch(model, tokenizer, texts):
#     device = next(model.parameters()).device
#     inputs = tokenizer(texts, return_tensors='pt', padding=False, truncation=True).to(device)
#     outputs = model(**inputs, labels=inputs['input_ids'])
#     return outputs.loss.item()

import torch
import torch.nn.functional as F

def clc_loss_batch(model, tokenizer, texts):
    device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    labels = inputs['input_ids'].roll(-1, dims=1)
    outputs = model(**inputs, labels=labels)

    return outputs.loss.item()


def get_loss_dataset(model, tokenizer, dataset, batch_size, name):
    device = next(model.parameters()).device
    dataset = TextExampleDataset(dataset)

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    total_loss = 0
    n = 0

    for batch in tqdm.tqdm(data_loader):
        if len(batch[name]) == 0:
            continue
        total_loss += clc_loss_batch(model, tokenizer, batch[name])
        n += len(batch[name])

    return total_loss / n
