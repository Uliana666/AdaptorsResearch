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
            # print(self.pre_layer.weight.shape, self.adapter.get_value().shape)
            self.pre_layer.weight -= self.alpha * self.adapter.get_value()

        self._enabled = True

    def enable_adapters(self, enable: bool = True):
        self._enabled = enable

    def forward(self, x: torch.Tensor):
        if not self._enabled:
            return self.pre_layer(x)
        
        
        # print(x.shape)
        x_1 = F.linear(x, self.pre_layer.weight)
        # print("-------------- OKOKOKOKK")
        x_2 = self.alpha * self.adapter(x)
        # print("--------------- MEOWMEOWMEOW")
        x = x_1 + x_2
        # print(x.shape)

        return x

    # def merge(self) -> nn.Linear:
    #     """
    #     merge may destruct GSOFTLinear structure. Do not use this layer after merging 
    #     """
    #     in_shape, out_shape = self.in_features, self.out_features
    #     W_0: torch.Tensor = self.pre_layer.weight.data

    #     if self.use_bias:
    #         raise NotImplementedError

    #     if self.is_left:
    #         I = torch.eye(in_shape, dtype=torch.float32, device=W_0.device)
    #         Q = self.gs_ort(I).to(dtype=W_0.dtype).transpose(0, 1)
    #         W = torch.mm(W_0, Q)
    #     else:
    #         I = torch.eye(out_shape, dtype=torch.float32, device=W_0.device)
    #         Q = self.gs_ort(I).to(dtype=W_0.dtype).transpose(0, 1) 
    #         W = torch.mm(Q, W_0)
        
    #     if self.scale:
    #         W.mul_(self.gsoft_s.unsqueeze(1))
        
    #     self.pre_layer.weight.data = W
    #     del W_0

    #     return self.pre_layer