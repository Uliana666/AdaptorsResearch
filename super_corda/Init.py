import math
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
import reprlib



class SCorDAInitialization(nn.Module):
    def __init__(self, 
                r: int, 
                init_strategy="lora", 
                base_tensor: torch.Tensor = None,
                ):


        super().__init__()

        self.adapter_A = nn.Parameter(base_tensor.new_empty(base_tensor.shape[1], r, dtype=torch.float32))
        self.adapter_B = nn.Parameter(base_tensor.new_empty(r, base_tensor.shape[0], dtype=torch.float32))

        self.r = r
        self.init_strategy = init_strategy

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.init_strategy == "lora":
            torch.nn.init.zeros_(self.adapter_B)
            torch.nn.init.normal_(self.adapter_A, mean=0.0, std=1.0)
        else:
            raise ValueError("I'm a cat")

    
    def forward(self, x):
        return torch.matmul(x, torch.matmul(self.adapter_A, self.adapter_B))
    
    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}(\n"
            f"  adapter_A=Parameter(shape={self.adapter_A.shape}, dtype={self.adapter_A.dtype}, requires_grad={self.adapter_A.requires_grad}),\n"
            f"  adapter_B=Parameter(shape={self.adapter_B.shape}, dtype={self.adapter_B.dtype}, requires_grad={self.adapter_B.requires_grad}),\n"
            f"  r={self.r},\n"
            f"  init_strategy='{self.init_strategy}'\n"
            f")"
        )
        return repr_str

    # def __repr__(self):
    #     return f"SCorDA(r={self.r}, init_strategy={self.init_strategy})"