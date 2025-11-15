from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
from src.data.tokenizer import BPETokenizer

class TextBPEBuffer(Dataset) : 
    def __init__(
        self,
        path: str, 
        tokenizer: BPETokenizer, 
        seq_length : 256
    ) : 
        super().__init__()
        self.seq_len = seq_length

        # Importing
        data = Path(path).read_text(encoding='utf-8')
        self.ids = torch.tensor(tokenizer.encode(text=data), dtype=torch.long)
        
        
    def __len__(self) : 
        return max(0, self.ids.numel() - self.seq_len - 1)
    
    def __getitem__(self, index : int):
        input = self.ids[index : index + self.seq_len]
        targets = self.ids[index + 1 : index + self.seq_len + 1]
        
        return input, targets
    
    def makeLoader(self, batch_size : int, shuffle = True) -> DataLoader : 
        dataset = self
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True # Drop the last batch if it is not divisble.
        )
        
        
        