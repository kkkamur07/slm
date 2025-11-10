import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class KVCache:
    """Standard KV cache for storing keys and values."""
    k: Optional[torch.Tensor] = None  # (B, H, T, d_head)
    v: Optional[torch.Tensor] = None  # (B, H, T, d_head)
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:  
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        
        return self.k, self.v
    
    @property
    def size(self) -> int:
        """Return current cache sequence length."""
        return self.k.size(2) if self.k is not None else 0
    
    def clear(self):
        """Reset cache."""
        self.k = None
        self.v = None


class RollingKVCache:
    """Rolling window KV cache with sink tokens (for long-context generation)."""
    
    def __init__(self, window: int, sink: int = 0):
        """
        Args:
            window: Recent window size to keep
            sink: Number of initial tokens to always keep
        """
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:  # Subsequent calls - concatenate
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        
        # Rolling window: discard old tokens except sink
        if self.k.size(2) > self.window + self.sink:
 
            sink_k = self.k[:, :, :self.sink, :]
            sink_v = self.v[:, :, :self.sink, :]
            

            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            
            self.k = torch.cat([sink_k, tail_k], dim=2)
            self.v = torch.cat([sink_v, tail_v], dim=2)
        
        return self.k, self.v
    
    @property
    def size(self) -> int:
        """Return current cache length."""
        return self.k.size(2) if self.k is not None else 0
    
    def clear(self):
        """Reset cache."""
        self.k = None
        self.v = None