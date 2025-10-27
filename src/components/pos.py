"""basics/pos.py : Implements the positional embeddings used in transformers."""

import math
import torch
import torch.nn as nn
import numpy as np

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
    
class ROPE : 
    def __init(self, input : torch.Tensor) -> torch.Tensor : 
        