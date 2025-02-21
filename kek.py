from lib import Models, Datasets, LossCalculator, Train
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

model = Models.LoadLLM("meta-llama/Llama-3.2-1B-Instruct")

math = Datasets.LoadMath()

print(model['model'])


model['model'] = Models.PrepareLoRA(model['model'], 16, 32, 0.1)


e = LossCalculator.CalcLoss(**math, max_length=1024, count=100, model=model['model'], tokenizer=model['tokenizer'])

print(e)

