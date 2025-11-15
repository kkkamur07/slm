from __future__ import annotations
import torch
import torch.nn as nn
from .block import TransformerBlock
from src.components.data.tokenizer import ByteTokenizer
import torch.nn.functional as F
from einops import rearrange
from utils import top_k_top_p_filtering

class GPTModern(nn.Module): 
                 
    def __init__(
        self, 
        vocab_size: int = 256,
        seq_length: int = 256,
        n_layer: int=4, 
        n_head: int=4,
        d_model: int=256, 
        dropout: float=0.0,
        use_rmsnorm: bool = True,
        # use_swiglu: bool = True, BY DEFAULT THIS IS TRUE. 
        rope: bool = True,
        max_pos: int = 4096,
        sliding_window: int | None = None,
        attention_sink: int = 0,
        n_kv_head: int | None = None
        ):
        super().__init__()
        self.seq_length = seq_length 
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_head=n_head,
                n_kv_head=n_kv_head,
                dropout=self.dropout.p,
                use_rmsnorm=use_rmsnorm,
                rope=rope,
                sliding_window=sliding_window,
                attention_sink=attention_sink,
                max_seq_length=max_pos
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.Identity() if use_rmsnorm else nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        
        
    def forward(self, idx : torch.Tensor, targets : torch.Tensor | None = None, kv_cache_list = None, start_pos: int = 0) : 
        B, T = idx.shape
        assert T <= self.seq_length, "Input sequence length shouldn't exceed models maxiumum sequence length"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) # No positional Embeddings here
        x = self.dropout(x)
        
        
        new_caches = []
        
        for i, block in enumerate(self.blocks) : 
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = block(x, kv_cache = kv_cache_list, start_pos=start_pos)
            new_caches.append(cache)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None : 
            logits_flat = rearrange(logits, 'b t v -> (b t) v')
            targets_flat = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss, new_caches
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 200,      
        temperature: float = 1.0,     
        top_k: int = 50,
        top_p: float | None = None,
        eos_id: int = 1,
        sliding_window: int | None = None, 
        attention_sink: int = 0,
    ):
        self.eval()
        idx = prompt
        kv = [None] * len(self.blocks)
        
        
        for _ in range(max_new_tokens) : 
            idx_cont = idx[:, -self.seq_length:] if kv[0] is None else idx[:,-1:]
            
            start_pos = 0 if kv[0] is None else kv[0].k.size(2)
            
            logits, _, kvs = self(idx_cont, kv_cache_list = kv, start_pos=start_pos)
            
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = top_k_top_p_filtering(logits=next, top_k=top_k, top_p=top_p)
            
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
            
            if eos_id is not None : 
                if ( next_id == eos_id ).all() : 
                    break
                
        return idx
    
    
    @torch.no_grad()
    def generate_nocache(
        self,
        prompt : torch.Tensor,
        max_new_tokens : int = 200,
        temperature : float =  1.0,
        top_k : int = 50,
        top_p : int | None = None,
        sliding_window : int | None = None,
        eos_id : int = 1,
        attention_sink : int = 0) : 

        self.eval()
        idx = prompt
        
        for _ in range(max_new_tokens) : 
            idx_cont = idx[:, -self.seq_length:]
            start_pos = idx.size(1) - idx_cont.size(1) # Calculate the starting position of the RoPe Cache. 
            logits, _, _ = self(idx_cont, kv_cache_list = None, start_pos=start_pos)
            
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = top_k_top_p_filtering(logits=next, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            topv , topk = torch.topk(probs, 10)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
            
            if (next_id == eos_id).all() : 
                break
            
        return idx
    
            
        
        
            
            
                   