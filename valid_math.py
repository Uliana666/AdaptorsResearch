from dataclasses import dataclass, field
from typing import Optional
from lib import Models, Datasets
import transformers
from transformers import Trainer
import numpy as np
from transformers import Trainer
from omegaconf import OmegaConf
from transformers import pipeline
from tqdm import tqdm


@dataclass
class InferenceArguments(transformers.TrainingArguments):
    path: Optional[str] = field(default="facebook/opt-125m")
    
    name: Optional[str] = field(default="facebook/opt-125m")
        
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
    
    count_examples: int = field(default=None, metadata={"help": "count of examples will be load from dataset"})
        
    start_token: int = field(default=None, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    
    name_dataset: str = field(default="common-reasoning", metadata={"help": "Chose: BoolQ, PIQA, SIQA, hellaswag, winogrande, ARC-E, ARC-C, OBQA"})
    
    config_path: str = field(default=None)


def valid():
    parser = transformers.HfArgumentParser(InferenceArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    adaptor_config = OmegaConf.load(script_args.config_path)

    model, tokenizer = Models.LoadFineTuneLLM(adaptor_config, script_args.name, script_args.path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000)
    
    dataset = Datasets.LoadCommonReasoning('validation', script_args.count_examples, 
                                                    script_args.name_dataset, script_args.seed_of_gen)
    positive = 0
    total = 0
    print('!', dataset)
    for sample in tqdm(dataset['dataset']):
        total += 1
        text = pipe(sample['text_wa_answer'])
        text = text[0]['generated_text']
        
        last_occurrence_index = text.rfind("The answer is:")

        if last_occurrence_index != -1:
            answer_start_index = last_occurrence_index + len("The answer is:")
            answer = text[answer_start_index:].strip().split()[0]
            print(sample['correct_answer'], answer)
            if sample['correct_answer'] == answer:
                positive += 1
                print('+')
    
        

if __name__ == "__main__":
    valid()