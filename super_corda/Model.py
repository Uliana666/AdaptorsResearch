import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel
from peft import PeftConfig, PeftType, PeftModel
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists
)
import os
import json
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    get_peft_model_state_dict
)
from super_corda.Scorda import SCorDALinear
from super_corda.Injector import inject_scorda

import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from transformers import PretrainedConfig
from transformers import PreTrainedModel




class SCorDAConfig(PeftConfig):
    model_type = "scorda"

    def __init__(self, r=8, alpha=16, dropout=0, init_strategy=None, samples=None, target_modules=None, **kwargs):
        super().__init__(peft_type="scorda", **kwargs)
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.init_strategy = init_strategy
        self.samples = samples
        self.target_modules = target_modules
    
    
    def __post_init__(self):
        self.peft_type = "SCORDA"
        

class SCorDAModel(torch.nn.Module):
    config_class = SCorDAConfig

    def __init__(self, config, model):
        
        super().__init__()
        # Ensure peft_config is a dict mapping adapter names to configs
        self.peft_config = config
        self.model = inject_scorda(model)
        
        
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config
        
# class SCorDAModel(PeftModel):
#     config_class = SCorDAConfig

#     def __init__(self, model, config, adapter_name=None):
#         print(config)  # For debugging
#         super().__init__(model, config['default'])
#         self.model = inject_scorda(config, model)
        
#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

#     def save_pretrained(self, save_directory):
#         super().save_pretrained(save_directory)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     config_path = os.path.join(pretrained_model_name_or_path, "config.json")
    #     config = SCorDAConfig.from_json_file(config_path)
    #     print(config)
    #     model = cls(config)
        
    #     state_dict = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"))
    #     model.load_state_dict(state_dict)
        
    #     return model

# class SCorDAModel(torch.nn.Module):
#     def __init__(self, config, model):
#         super().__init__()
#         self.peft_config = config
#         self.model = model
#         self._find_and_replace()
#         self.forward = self.model.forward

#     def _find_and_replace(self):
#         self.model = inject_scorda(self.peft_config, self.model)

#     def _get_submodules(self, key):
#         parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
#         target_name = key.split(".")[-1]
#         target = self.model.get_submodule(key)
#         return parent, target, target_name


#     def __getattr__(self, name: str):
#         """Forward missing attributes to the wrapped module."""
#         try:
#             return super().__getattr__(name)  # defer to nn.Module's logic
#         except AttributeError:
#             return getattr(self.model, name)

#     @property
#     def modules_to_save(self):
#         return None

#     def get_peft_config_as_dict(self, inference: bool = False):
#         config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
#         if inference:
#             config["inference_mode"] = True
#         return config
    
#     def save_pretrained(self, save_directory):
#         os.makedirs(save_directory, exist_ok=True)
        
#         weights_path = os.path.join(save_directory, "pytorch_model.bin")
#         torch.save(self.model.state_dict(), weights_path)

#         config_path = os.path.join(save_directory, "config.json")
#         with open(config_path, "w") as f:
#             json.dump(self.get_peft_config_as_dict(), f)
            
#     @classmethod
#     def from_pretrained(cls, load_directory, model):
#         config_path = os.path.join(load_directory, "config.json")
#         with open(config_path, "r") as f:
#             config_data = json.load(f)

#         config = SCorDAConfig(**config_data)

#         model_instance = cls(config, model)

#         weights_path = os.path.join(load_directory, "pytorch_model.bin")
#         model_instance.model.load_state_dict(torch.load(weights_path))

#         return model_instance