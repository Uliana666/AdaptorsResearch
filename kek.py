from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Загрузка модели и токенизатора
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Установка pad_token для совместимости
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Загрузка датасета
dataset = load_dataset("meta-math/MetaMathQA", split='train[:20]')

# Функция для форматирования и токенизации
def format_and_tokenize(examples):
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['query'], examples['response'])]
    return tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )

# Токенизация данных
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=dataset.column_names
)

# Создание DataCollator для языкового моделирования
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Настройка параметров оценки
training_args = TrainingArguments(
    output_dir="./results",
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=1, 
    fp16=True,
    report_to="none",
    eval_accumulation_steps=1
)

print(tokenized_dataset)

# Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=tokenized_dataset,  # Используем подмножество для демонстрации
    tokenizer=tokenizer,
)

# Вычисление loss
eval_results = trainer.evaluate()
print(f"Evaluation Loss: {eval_results['eval_loss']:.4f}")