"""1.2 This module implements the multiHead attention"""

import math
import torch 
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module) : 
    def __init__(self, heads : int, d_model : int, d_v :int | None,  dropout : float = 0.0, trace_shapes : bool = False) : 
        super().__init__()
        
        self.heads = heads
        assert d_model % self.heads == 0 
        
        #? Dimensions
        self.d_model = d_model
        
        if (d_v is None) :
            self.d_v = d_model // heads
        else :
            self.d_v = d_v # Can be any arbitrary number
            
        self.d_k = d_model // heads
        
        self.dropout = nn.Dropout(dropout)
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
                
        q = self.q_proj(input)  # (B, T, heads * d_k)
        k = self.k_proj(input)  # (B, T, heads * d_k)
        v = self.v_proj(input)  # (B, T, heads * d_v)
        
        # Changing the shapes B, H, T, D
        q = q.view(batch, tokens, self.heads, self.d_k).transpose(1,2)
        k = k.view(batch, tokens, self.heads, self.d_k).transpose(1,2)
        v = v.view(batch, tokens, self.heads, self.d_v).transpose(1,2)
        
        if self.trace_shapes : # Kind of like verbose 
            print(f"Shape Q : {q.shape}, Shape K : {k.shape}, Shape V : {v.shape}")
            
        context = F.scaled_dot_product_attention(
            q,
            k,
            v, 
            attn_mask= None, 
            dropout_p= self.dropout if self.training else 0.0, 
            is_causal= True
        )
        
        out = context.transpose(1,2).contiguous().view(batch, tokens, self.heads * self.d_v) 
        # B, T, H, d_v -> B, T, H * d_v
        out = self.out_proj(out) # Concatenation of the heads
        
        return out
        
        
        
        
        
            
        
        
        
        
