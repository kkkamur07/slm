"""The power of RMSNorm is that it can learn the normalization"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module) : 
    """
    y = x * g / rms(x), rms(x) = sqrt(mean(x^2) + eps)
    g -> assigns the weights which can be learned
    """
    
    def __init__(self, dim: int, eps: float = 1e-8) : 
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # Normalization is being learnt now
        
    def forward(self, input:torch.Tensor) -> torch.Tensor : 
        # One sequence of instruction for GPU friendlyNess
        rms = input.pow(2).mean(dim =-1,keepdim=True).add(self.eps).sqrt()
        return (input / rms) * self.weight
    


