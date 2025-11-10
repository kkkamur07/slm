"""basics/pos.py : Implements the positional embeddings used in transformers."""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

#? nn.Module needs to implement an forward() method.
class LearnedPositionalEncodings(nn.Module) : 
    def __init__(self, max_len : int, d_model : int) : 
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        # These embeddings will be added to the actual embeddings so needs to be of the same dimension. 
        
    def forward(self, input : torch.Tensor) : 
        B, length, dimensions = input.shape # Batch, Length, Dimensions
        position = torch.arange(length, device=input.device)
        pos_emb = self.emb(position) # Length, d_model 
        return input + pos_emb.unsqueeze(0) # Adding a new dimension to the first position. 
    
class SinusodialPositionalEmbeddings(nn.Module) : 
    def __init__(self, max_len : int, d_model : int) : 
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        self.pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000) / d_model))
        
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
        
        self.register_buffer("pe", torch.from_numpy(self.pe))
        
    def forward(self, input : torch.Tensor) : 
        B, length, dimensions = input.shape
        return input + self.pe[:length].unsqueeze(0) # only required positional embeddings till length. 
    
  
class ROPE(nn.Module) : 
    def __init__(self, head_dim : int, max_seq_len : int, base : float = 10000, device: Optional[torch.device] = None) : 
        super().__init__()
        assert head_dim % 2 == 0 # why only even head dim ? 
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        self._build_cache(max_seq_len, device)
        
    def _build_cache(self, max_len : int, device : Optional[torch.device] = None) : 
       self.max_seq_len = max_len
       
       # Compute the inverse frequencies
       inv_freq = 1.0 / (
           self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)
           )
       
       # Positions
       positions = torch.arange(max_len,device=device,dtype = torch.float32)
       
       # Outer product
       freqs = torch.outer(positions, inv_freq)
       
       self.register_buffer("cos_cached", freqs.cos(), persistent=False)
       self.register_buffer("sin_cached", freqs.sin(), persistent=False)
       
    def forward(self, x : torch.Tensor, seq_len : Optional[int] = None, start_pos : int = 0) -> torch.Tensor : 
        if seq_len is None : 
            seq_len = x.size(2) # 3 elements
            
        end_pos = start_pos + seq_len
            
        if end_pos > self.max_seq_len : 
            self._build_cache(max(end_pos, 2 * self.max_seq_len), device=x.device)
            
        cos = self.cos_cached[start_pos:end_pos]
        sin = self.sin_cached[start_pos:end_pos]
        
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        return self._apply_rotation(x, cos, sin)
    
    def _apply_rotation(self, x : torch.Tensor, cos : torch.Tensor, sin : torch.Tensor) : 
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        x_rotated = torch.empty_like(x)
        x_rotated[...,0::2] = x_even_rot
        x_rotated[...,1::2] = x_odd_rot
        
        return x_rotated
    
    def extra_repr(self):
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, base={self.base}"