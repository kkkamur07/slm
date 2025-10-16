"""1.2 This module implements the multiHead attention"""

import math
import torch 
from torch import nn
import torch.nn.functional as F
from src.basics.mask import causal_mask

class MultiHeadAttention(nn.Module) : 
    def __init__(self, heads : int, d_model : int, d_v :int,  dropout : float = 0.0, trace_shapes : bool = False) : 
        super().__init__()
        
        self.heads = heads
        assert d_model % self.heads == 0
        
        #? Dimensions
        self.d_model = d_model
        self.d_v = d_v # Can be any arbitrary number
        self.d_k = d_model // heads
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.trace_shapes = trace_shapes
        
        # Projections: q/k project to heads * d_k (== d_model), v projects to heads * d_v
        self.q_proj = nn.Linear(d_model, heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, heads * self.d_v, bias=False)
        
        # For the concatenation of the heads
        self.out_proj = nn.Linear(heads * self.d_v, d_model, bias=False)
    
        
    def forward(self, input : torch.Tensor) : 
        batch, tokens, dimensions = input.shape
        assert dimensions == self.d_model
        scale = math.sqrt(self.d_k)
        
        q = self.q_proj(input)  # (B, T, heads * d_k)
        k = self.k_proj(input)  # (B, T, heads * d_k)
        v = self.v_proj(input)  # (B, T, heads * d_v)
        
        # Changing the shapes B, H, T, D
        q = q.view(batch, tokens, self.heads, self.d_k).transpose(1,2)
        k = k.view(batch, tokens, self.heads, self.d_k).transpose(1,2)
        v = v.view(batch, tokens, self.heads, self.d_v).transpose(1,2)
        
        if self.trace_shapes : # Kind of like verbose 
            print(f"Shape Q : {q.shape}, Shape K : {k.shape}, Shape V : {v.shape}")
            
        # Scores : B, H, T, T
        scores = (q @ k.transpose(-2, -1) ) / scale
        mask = causal_mask(tokens, device=input.device) # So that it cannot attend to the future
        scores = scores.masked_fill(mask, float("-inf"))
        
        attention = F.softmax(scores, dim=-1)
        attention_regularized = self.dropout(attention)
        context = attention_regularized @ v # B, H, T, d_v
        
        # Final Processing
        out = context.transpose(1,2).contiguous().view(batch, tokens, self.heads * self.d_v) 
        # B, T, H, d_v -> B, T, H * d_v
        out = self.out_proj(out) # Concatenation of the heads
        
        return out
        
        
        
        
        
            
        
        
        
        
