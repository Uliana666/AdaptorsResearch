import tqdm
import torch

def tokenize_function(tokenizer, examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=32, return_tensors='pt')

def compute_loss(model, tokenizer, example):
    inputs = tokenize_function(tokenizer, example)
    device = next(model.parameters()).device
    
    inputs['input_ids'] = inputs['input_ids'].to(device)
    
    inputs['labels'] = inputs['input_ids'].clone().to(device)
    # inputs['labels'][:, :-1] = inputs['input_ids'][:, 1:].clone().to(device)
    
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    
    outputs = model(inputs['input_ids'], labels=inputs['labels'], attention_mask=inputs['attention_mask'])
    return outputs.loss

def train_model(model, dataset, optimizer, tokenizer, epochs=1):
    model.train()
    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        k = 0
        

        for example in dataset:
            # print(example)
            k += 1

            loss = compute_loss(model, tokenizer, example)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            if k % 80 == 0:
                print(total_loss / k)
                total_loss = 0
                k = 0

        optimizer.zero_grad()
