import tqdm
import torch


def compute_loss(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    
    labels = inputs['input_ids'].roll(-1, dims=1)
    # outputs = model(inputs['input_ids'], labels=inputs['labels'], attention_mask=torch.triu(inputs['attention_mask']))
    # print(['attention_mask'])
    mask = torch.triu(inputs['attention_mask'])
    outputs = model(inputs['input_ids'], labels=labels, attention_mask=mask)
    loss = outputs.loss
    return loss

def train_model(model, tokenizer, dataset, name, optimizer, accumulate_steps=1, epochs=1):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n = 0

        for example in tqdm.tqdm(dataset):
            if len(example[name]) == 0:
                continue

            n += 1

            optimizer.zero_grad()

            loss = compute_loss(model, tokenizer, example[name]) / accumulate_steps

            total_loss += loss.item()
            loss.backward()

            if (n + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(total_loss)

        average_loss = total_loss / n if n > 0 else 0
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")
        optimizer.zero_grad()
