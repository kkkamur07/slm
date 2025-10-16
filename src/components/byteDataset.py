from __future__ import annotations
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np

class ByteDataset(DataLoader) : 
    def __init__(self, path : str, block_size : int = 256, split : float = 0.9, val : bool = False): 
        super().__init__()
        self.block_size = block_size
        self.path = path
        self.split = split
        self.val = val
        
        # For efficiency reasons using `numpy`
        data = np.frombuffer(Path(self.path).read_bytes(), dtype=np.uint8)
        self.data = torch.from_numpy(data).long()
        self.n = int(len(data) * self.split)
        
        self.train_data = self.data[:self.n]
        self.val = self.data[self.n:]
        
    def __len__(self) -> int : 
        return len(self.data)
    
    def __getitem__(self, idx : int) : 
        # This is the full paradigm of self supervised learning. 
        if self.val == True : 
            x = self.val[idx : idx + self.block_size]
            y = self.val[idx + 1 : idx + self.block_size + 1]
        else : 
            x = self.train_data[idx : idx + self.block_size]
            y = self.train_data[idx + 1 : idx + self.block_size + 1]
            
        return x, y
    
    