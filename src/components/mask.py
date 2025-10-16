"""1.3 Main job is to return the attention mask -> upper triangular matrix"""

import torch 

def causal_mask(T : int, device = None) : 
    m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return m.view(1,1,T,T) # ensuring compatibility with B, H, T, T