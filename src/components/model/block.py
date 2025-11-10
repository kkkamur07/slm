import torch 
import torch.nn as nn
from .attention import CausalSelfAttention, GroupQueryAttention
from .ffn import FeedForwardNetwork
from .rms_norm import RMSNorm
from ..activation import SwiGLU

class TransformerBlock(nn.Module) : 
    def __init__(
        self,
        d_model : int,
        n_head : int,
        n_kv_head : int | None = None, 
        dropout : float = 0.0,
        use_rmsnorm : bool = True,
        rope : bool = True,
        seq_len : int = 4096,
        sliding_window : int | None = None,
        attention_sink : int = 0
        ) : 
        super().__init__()
        norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.layer1 = norm(d_model)
        self.attention = GroupQueryAttention(
            d_model=d_model,
            n_heads=n_head,
            n_kv_heads=n_kv_head,
            dropout=dropout,
            rope=rope,
            sliding_window=sliding_window,
            max_seq_length=seq_len,
            trace_shapes=False,
            attention_sink=attention_sink
        )
        self.layer2 = norm(d_model)
        self.ffn = SwiGLU(dim=d_model, mult=4, dropout=dropout)
    
    def forward(self, input : torch.Tensor, kv_cache = None, start_pos : int = 0) : 
        a, kv_cache = self.attention(self.layer1(input), kv_cache=kv_cache, start_pos=start_pos)
        input += a # Residual connection
        input += self.ffn(self.layer2(input))
        
        return input, kv_cache
    
    
    