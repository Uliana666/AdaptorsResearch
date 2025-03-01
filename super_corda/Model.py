import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel
from peft import PeftConfig
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    get_peft_model_state_dict
)
from Scorda import SCorDALinear


class SCorDAConfig(PeftConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.peft_type = "SCorDA"
        self.some_parameter = kwargs.get('some_parameter', 42)


    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        return PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)


class SCorDAModel(BaseTuner):

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        self._adapter_name_prefix = 'scorda'
        self.config = config

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def merge_and_unload(self, progressbar: bool = False, safe_merge: bool = False):
        return self._unload_and_optionally_merge(merge=True, progressbar=progressbar, safe_merge=safe_merge)

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _check_target_module_exists(self, peft_config: SCorDAConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(self, scorda_config: SCorDAConfig, adapter_name: str, target: nn.Module, target_name: str, parent: nn.Module, current_key: str) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        
        kwargs = {
            'r': scorda_config.get('r', None),
            'orthogonal': scorda_config.orthogonal,
            'method': scorda_config.method,
            'alpha': scorda_config.alpha,
            'init_strategy': scorda_config.init_strategy,
        }
        
        if isinstance(target, SCorDALinear):
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(scorda_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module: SCorDALinear, child):
        setattr(parent, child_name, new_module)

    @staticmethod
    def _create_new_module(scorda_config, adapter_name, target, **kwargs):
        out_f, in_f = target.weight.shape
        scorda_linear = SCorDALinear(
            pre_layer=target,
            in_features=in_f, out_features=out_f,
            **kwargs
        )
        return scorda_linear

    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        for name, param in model.named_parameters():
            if self._adapter_name_prefix not in name:
                param.requires_grad = False