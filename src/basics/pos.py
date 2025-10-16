"""basics/pos.py : Implements the positional embeddings used in transformers."""

import math
import torch
import torch.nn as nn

#? nn.Module needs to implement an forward() method.
class LearnedPositionalEncodings(nn.Module) : 
    def __init__(self, max_len : int, d_model : int) : 
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        # These embeddings will be added to the actual embeddings so needs to be of the same dimension. 
        
    def forward(self, input : torch.Tensor) : 
        B, length, dimensions = input.shape # Batch, Length, Dimensions
        position = torch.arange(length, device=input.device)
        pos_emb = self.emb(position) # Length, Dimensions
        return input + pos_emb.unsqueeze(0) # Adding a new dimension to the first position. 
    
class SinusodialPositionalEmbeddings(nn.Module) : 
    def __init__(self, max_len : int, d_model : int) : 
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model) # placeholders for the positional embeddings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # For broadcasting
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
        
    def forward(self, input : torch.Tensor) : 
        B, length, dimensions = input.shape
        return input + self.pe[:length].unsqueeze(0) # only required positional embeddings till length. 
    
    