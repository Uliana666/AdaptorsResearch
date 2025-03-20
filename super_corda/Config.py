from dataclasses import dataclass
from typing import List
from lib import Datasets
from transformers import Trainer
import torch
import tempfile
import os
from super_corda.Injector import inject_scorda, prepare_get_samples, after_get_samples

@dataclass
class SCorDAConfig:
    r: int
    alpha: int
    target_modules: List[str]
    init_strategy: str = None
    samples: str = None
    
    
def get_peft_model(model, config, args=None, tokenizer=None, logs=None):
    if config.init_strategy in  ["corda", "scorda", "scorda_svf"]:
        model, hooks = prepare_get_samples(model, config)
        data = Datasets.LoadCommonReasoning('train', None, config.samples)
    
        dataset, data_collator = Datasets.PrepareDataset(**data, args=args, tokenizer=tokenizer, desc="peft")

        direct = args.output_dir
        
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = temp_dir
            args.logging_dir = temp_dir

            trainer = Trainer(
                model=model,
                args=args,
                data_collator=data_collator,
                eval_dataset=dataset,
                tokenizer=tokenizer,
                preprocess_logits_for_metrics=Datasets.preprocess_logits_for_metrics
            )

            trainer.evaluate()
    
            after_get_samples(model, config, hooks)
    
        args.output_dir = direct
    
    return inject_scorda(config, model, logs)
