import numpy as np
import tqdm
import torch
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm


# def clc_loss_batch(model, tokenizer, texts):
#     device = next(model.parameters()).device
#     inputs = tokenizer(texts, return_tensors='pt', padding=False, truncation=True).to(device)
#     outputs = model(**inputs, labels=inputs['input_ids'])
#     return outputs.loss.item()

import torch
import torch.nn.functional as F

def clc_loss_batch(model, inputs):
    outputs = model(**inputs)
    return outputs.loss.item()


def get_loss_dataset(model, dataset):
    device = next(model.parameters()).device

    total_loss = 0
    n = 0

    for batch in tqdm.tqdm(data_loader):
        if len(batch[name]) == 0:
            continue
        total_loss += clc_loss_batch(model, tokenizer, batch[name])
        n += len(batch.shape[0])

    return total_loss / n
