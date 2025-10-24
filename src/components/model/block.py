import torch 
import torch.nn as nn
from .multiheadAttention import CausalSelfAttention
from .ffn import FeedForwardNetwork

class Block(nn.Module) : 
    def __init__(self, d_model : int, heads : int, dropout : float, d_v : int | None) : 
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout_p = dropout
        self.dropout = dropout
        self.d_v = d_v
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model=d_model, d_v=d_v, heads=heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, mul_factor=4, dropout=dropout) # Setting the multiplication factor as 4
        
    def forward(self, input : torch.Tensor) : 
        # Attention Block
        attn_out = self.attention(self.ln1(input))
        attn_out = attn_out + input # Residual Connection
        
        # Feed Forward Network Block
        ffn_out = self.ffn(self.ln2(attn_out))
        ffn_out = ffn_out + attn_out # Residual Connection
        
        return ffn_out