"""Lean implementation of the ByteTokenizer"""

from __future__ import annotations # Gives you cleaner type hints
import torch
import numpy as np 



class ByteTokenizer :

    def __init__(self) : 
        self.encoding = "utf-8"  #! Check for whether indic languages this could be a problem ?
        self.vocabSize = 256
        
    def encode(self, s: str) -> torch.Tensor : 
        encoded = np.frombuffer(s.encode(self.encoding)) # STRING TO LIST
        return torch.from_numpy(encoded).long()
    
    def decode(self, ids : np.ndarray) -> str:
        if  isinstance(ids, torch.Tensor) : 
            ids = ids.cpu().numpy() #! is this efficient ? 
        return bytes(ids).decode(self.encoding, errors="replace") # LIST TO BYTES TO STRING
    
    def __len__(self) -> int : 
        return self.vocabSize
    
    