from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

class ByteDataset(Dataset) : 
    def __init__(self, path : str, seq_len : int = 256, split : float = 0.9, train : bool = False): 
        super().__init__()
        self.seq_len = seq_len
        self.path = path
        self.split = split
        self.train = train
        
        # For efficiency reasons using `numpy`
        data = np.frombuffer(Path(self.path).read_bytes(), dtype=np.uint8)
        data = data.copy()
        data = torch.from_numpy(data).long()
        
        n = int(len(data) * self.split)
        self.data = data[:n] if train else data[n:]
        
    def __len__(self) -> int : 
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx : int) : 
        # This is the full paradigm of self supervised learning. 
        input = self.data[idx : idx + self.seq_len]
        target = self.data[idx + 1 : idx + self.seq_len + 1]
        
         # Safety check: ensure we got the right size
        assert len(input) == self.seq_len, f"Input length {len(input)} != block_size {self.block_size}"
        assert len(target) == self.seq_len, f"Target length {len(target)} != block_size {self.block_size}"
        
        return input, target
    
    