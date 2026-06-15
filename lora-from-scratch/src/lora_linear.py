import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self,in_features,out_features,r=16,alpha=32):
        super().__init__()
        self.weight=nn.Parameter(
            torch.randn(
                out_features,
                in_features
            )
        )
        self.weight.requires_grad=False
        self.A=nn.Parameter(
            torch.randn(
                r,
                in_features
            )*0.01
        )
        self.B=nn.Parameter(
            torch.zeros(
                out_features,
                r
            )
        )
        self.scaling=alpha/r
    def forward(self,x):
        base=x @ self.weight.T
        lora=x @ self.A.T
        lora=lora @ self.B.T
        lora=self.scaling * lora

        return base + lora