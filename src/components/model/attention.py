"""1.2 This module implements the multiHeaded attention & group query attention"""

import math
import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..pos import ROPE
from .kvcache import KVCache

class CausalSelfAttention(nn.Module) : 
    def __init__(self, heads : int, d_model : int, d_v :int | None = None,  dropout : float = 0.0, trace_shapes : bool = False) : 
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
        
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
        
        # Projections: q/k project to heads * d_k (== d_model), v projects to heads * d_v
        self.q_proj = nn.Linear(d_model, heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, heads * self.d_v, bias=False)
        
        # For the concatenation of the heads
        self.out_proj = nn.Linear(heads * self.d_v, d_model, bias=False)
    
        
    def forward(self, input : torch.Tensor) -> torch.Tensor : 
        batch, tokens, dimensions = input.shape
        assert dimensions == self.d_model
                
        q = self.q_proj(input)  # (B, T, heads * d_k)
        k = self.k_proj(input)  # (B, T, heads * d_k)
        v = self.v_proj(input)  # (B, T, heads * d_v)
        
        # Changing the shapes B, H, T, D
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.heads)  # (B, H, T, d_k)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.heads)  # (B, H, T, d_k)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.heads)  # (B, H, T, d_v)
        
        if self.trace_shapes : # Kind of like verbose 
            print(f"Shape Q : {q.shape}, Shape K : {k.shape}, Shape V : {v.shape}")
            
        context = F.scaled_dot_product_attention(
            q,
            k,
            v, 
            attn_mask= None, 
            dropout_p= self.dropout_p if self.training else 0.0, 
            is_causal= True
        )
        
        out = rearrange(context, 'b h t d -> b t (h d)').contiguous() 
        out = self.out_proj(out) # Concatenation of the heads
        
        return out
        

""" Here we will be grouping the queries 

INPUT : (B, T, d_model)


"""
class GroupQueryAttention(nn.Module) : 
    def __init__(
        self, 
        d_model : int, 
        n_heads : int,
        n_kv_heads : int | None = None, 
        dropout : float = 0.0,
        rope : bool = True,
        max_seq_length : int = 4096,
        sliding_window : int | None = None,
        trace_shapes : bool = False,
        attention_sink : int = 0,  
        ):
        
        super().__init__()
        
        assert d_model % n_heads == 0, "We can only split the embeddings if the heads are available."
        self.n_heads = n_heads
        self.d_model = d_model
        
        
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "We need the a group of queries to attend the KV, for that they need to be divisible."
        
        
        self.d_head = self.d_model // self.n_heads 
        self.group_size = self.n_heads // self.n_kv_heads
        self.seq_len = max_seq_length
                
        # Projection matrices : 
        self.wq = nn.Linear(d_model, self.n_heads * self.d_head, bias = False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias = False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias = False)
        self.proj = nn.Linear(d_model, d_model, bias = False) # outwards projection
        self.dropout = nn.Dropout(dropout)
        
        self.use_rope = rope
        if self.use_rope: 
            self.rope = ROPE(
                head_dim=self.d_head,
                max_seq_len=self.seq_len,
                base=10000
            )
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink
        
        # Verbosity
        self.trace_shapes = trace_shapes
        
            
    def forward(
        self, 
        input : torch.Tensor,
        kv_cache : KVCache | None = None, 
        start_pos : int = 0,
        use_cache : bool = False,
        ) -> tuple[torch.Tensor, KVCache | None]: 
        
        batch, tokens, d_model = input.shape
        assert d_model == self.d_model
        
        
        # INPUT : B,T, d_model -> B,T, n_heads * d_head -> B, n_q / n_k, T, d_head
        q = rearrange(self.wq(input), "b t (h d) -> b h t d", h = self.n_heads)
        k = rearrange(self.wk(input), "b t (k d) -> b k t d", k = self.n_kv_heads)
        v = rearrange(self.wv(input), "b t (v d) -> b v t d", v = self.n_kv_heads)
        
        if self.trace_shapes : 
            print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")
            
        # Cached Internally   
        if self.use_rope : 
            q = self.rope(q, seq_len = tokens, start_pos=start_pos)
            k = self.rope(k, seq_len = tokens, start_pos=start_pos)
            
        if use_cache : 
            if kv_cache is None : 
                kv_cache = KVCache()
            k, v = kv_cache.update(k, v)
        
        if self.sliding_window is not None and k.size(2) > (self.sliding_window + self.attention_sink) : 
            
            # Only keep required tokens from start.
            sink_k = k[:, :, :self.attention_sink, :]
            sink_v = v[:, :, :self.attention_sink, :]
            
            # Keep sliding window tokens from end
            window_k = k[:, :, -self.sliding_window:, :]
            window_v = v[:, :, -self.sliding_window:, :]
            
            # Concatenate -> Just discard the middle tokens
            k = torch.cat([sink_k, window_k], dim = 2).contiguous()
            v = torch.cat([sink_v, window_v], dim = 2).contiguous()
            
            if use_cache : 
                kv_cache.k = k
                kv_cache.v = v
            
        
        # If they are not equal then we do the grouping
        if self.n_kv_heads != self.n_heads : 
            k = repeat(k, "b h t d -> b (h group) t d", group = self.group_size)
            v = repeat(v, "b h t d -> b (h group) t d", group = self.group_size)
            
        is_causal = (kv_cache is None) and (not use_cache)
        
        context = F.scaled_dot_product_attention(
            q,k,v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Merge heads
        context = rearrange(context, "b h t d -> b t (h d)").contiguous()
        output = self.proj(context) # B T d_model
        
        return output, kv_cache if use_cache else None
        
        
        
        
        
        
        