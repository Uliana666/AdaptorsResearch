import torch
import torch.nn as nn
import torch.nn.functional as F

import super_corda.Linear as Linear
import torch

from peft.tuners.tuners_utils import BaseTunerLayer


class SCorDALayer(nn.Module, BaseTunerLayer):
    def __init__(
            self,
            pre_layer: nn.Module,
            in_features: int,
            out_features: int,
            r: int,
            init_strategy: str = 'lora',
            alpha: int = 1,
            X = None,
        ):

        super().__init__()

        self.pre_layer = pre_layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = r / alpha
        
        base_tensor = pre_layer.weight
        self.adapter = Linear.SCorDALinear(r, init_strategy=init_strategy, base_tensor=base_tensor, X=X)
        with torch.no_grad():
            self.pre_layer.weight -= self.alpha * self.adapter.get_value()

        self._enabled = True

    def enable_adapters(self, enable: bool = True):
        self._enabled = enable

    def forward(self, x: torch.Tensor):
        if not self._enabled:
            return self.pre_layer(x)
        
        x_1 = F.linear(x, self.pre_layer.weight)
        x_2 = self.alpha * self.adapter(x)
        x = x_1 + x_2

        return x