import torch 
from torch import nn 

class RMSnorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.eps = eps 
    def forward(self,x):
        input_dtype = x.dtype 
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1,keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype) * self.weight

