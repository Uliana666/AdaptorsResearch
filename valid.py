from dataclasses import dataclass, field
from typing import Optional
from lib import Models, Datasets
import transformers
from transformers import Trainer
import numpy as np
from transformers import Trainer
from omegaconf import OmegaConf


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


def valid():
    parser = transformers.HfArgumentParser(InferenceArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    adaptor_config = OmegaConf.load(script_args.config_path)

    model, tokenizer = Models.LoadFineTuneLLM(adaptor_config, script_args)

    print('MODEL READY')
        
    print(model)

    common_reasoning = Datasets.LoadCommonReasoning('validation', script_args.count_examples, 
                                                    script_args.name_dataset, script_args.seed_of_gen)
    
    dataset, data_collator = Datasets.PrepareDataset(**common_reasoning, args=script_args, 
                                                    tokenizer=tokenizer, desc="train")


    trainer = Trainer(
        model=model,
        args=script_args,
        data_collator=data_collator,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=Datasets.preprocess_logits_for_metrics
    )
    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == "__main__":
    valid()