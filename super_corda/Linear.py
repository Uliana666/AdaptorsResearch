import math
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
import reprlib

from lib import Compressors

# RANK = 256

class SCorDALinear(nn.Module):
    def __init__(self, 
                r: int, 
                init_strategy="lora", 
                base_tensor: torch.Tensor = None,
                X: torch.Tensor = None,
                ):


        super().__init__()

        self.adapter_A = nn.Parameter(base_tensor.new_empty(base_tensor.shape[1], r, dtype=torch.float32))
        self.adapter_B = nn.Parameter(base_tensor.new_empty(r, base_tensor.shape[0], dtype=torch.float32))
        
        # self.adapter_v = nn.Parameter(base_tensor.new_empty(RANK, 1, dtype=torch.float32))
        # self.adapter_U = nn.Parameter(base_tensor.new_empty(base_tensor.shape[1], RANK, dtype=torch.float32))
        # self.adapter_Vt = nn.Parameter(base_tensor.new_empty(RANK, base_tensor.shape[0], dtype=torch.float32))


        self.r = r
        self.init_strategy = init_strategy
        self.X = X

        self.reset_parameters(base_tensor)

    @torch.no_grad()
    def reset_parameters(self, base_tensor):
        if self.init_strategy == "lora":
            torch.nn.init.zeros_(self.adapter_B)
            # torch.nn.init.normal_(self.adapter_A, mean=0.0, std=1.0)
            torch.nn.init.kaiming_uniform_(self.adapter_A, a=math.sqrt(5))
            
        elif self.init_strategy == "pissa":
            U, VT = Compressors.PISSA(base_tensor, self.r)
            self.adapter_A.copy_(VT.T)
            self.adapter_B.copy_(U.T)
            
        elif self.init_strategy == "corda":
            B, A = Compressors.CORDA(base_tensor, self.X, self.r)
            del self.X
            self.adapter_A.copy_(A.T)
            self.adapter_B.copy_(B.T)
            
        elif self.init_strategy == "scorda":
            B, A = Compressors.SCORDA(base_tensor, self.X, self.r)
            del self.X
            self.adapter_A.copy_(A.T)
            self.adapter_B.copy_(B.T)
            
        # elif self.init_strategy == "scorda_svf":
        #     # B, A = Compressors.SCORDA(base_tensor, self.X, self.r)
        #     B, A = Compressors.SCORDA(base_tensor, self.X, self.r)
        #     del self.X
        #     self.adapter_A.copy_(A.T)
        #     self.adapter_B.copy_(B.T)
            
        #     # self.adapter_v.fill_(0)
        #     self.adapter_v.fill_(1)
        #     with torch.no_grad():
        #         U, S, Vt = Compressors.SVF(base_tensor, RANK)
        #         S = torch.sqrt(S)

            
        #     print(U.shape, S.shape, Vt.shape, "meow")
        #     print(self.adapter_Vt.shape, self.adapter_U.shape)
        #     print((U @ torch.diag_embed(S)).shape)
        #     self.adapter_Vt.copy_((U @ torch.diag_embed(S)).T)
        #     self.adapter_U.copy_(Vt.T)
            
        #     self.adapter_Vt.requires_grad = False
        #     self.adapter_U.requires_grad = False
        #     # print("SETTTTT")
            
        else:
            raise ValueError("I'm a cat")
        
    def get_value(self):
        if self.init_strategy != "scorda_svf":
            return (self.adapter_A @ self.adapter_B).T
        return (self.adapter_A @ self.adapter_B).T + (self.adapter_U @ torch.diag_embed(self.adapter_v[:, 0]) @ self.adapter_Vt).T

    
    def forward(self, X):
        Y = X @ (self.adapter_A @ self.adapter_B)
        if self.init_strategy == "scorda_svf":
            # print("HERE")
            Y += X @ (self.adapter_U @ torch.diag_embed(self.adapter_v[:, 0]) @ self.adapter_Vt)
        return Y
    
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
