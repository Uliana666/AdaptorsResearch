from dataclasses import dataclass, field
from typing import List
from lib import Datasets, Globals
from transformers import Trainer
import torch
from super_corda.Injector import inject_scorda, prepare_get_samples, after_get_samples

@dataclass
class SCorDAConfig:
    r: int
    alpha: int
    target_modules: List[str]
    dropout: int = 0
    init_strategy: str = None
    samples: str = None
    
    
def get_peft_model(model, config, args=None, tokenizer=None, logs=None):
    if config.init_strategy == "corda" or config.init_strategy == "scorda" or config.init_strategy == "scorda_svf":
        model, hooks = prepare_get_samples(model, config)
        data = Datasets.LoadCommonReasoning('train', None, config.samples)
    
        dataset, data_collator = Datasets.PrepareDataset(**data, args=args, tokenizer=tokenizer, desc="peft")

        direct = args.output_dir
        args.output_dir = "./tmp_trainer"
        args.logging_dir = "./tmp_trainer"
        
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            eval_dataset=dataset,
            tokenizer=tokenizer,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
    
        trainer.evaluate()
        after_get_samples(model, config, hooks)
        args.output_dir = direct
    
    return inject_scorda(config, model, logs)


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.stack([torch.argmax(logit, dim=-1) for logit in logits])
    return pred_ids, labels