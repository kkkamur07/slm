"""Lean implementation of the ByteTokenizer"""

from __future__ import annotations # Gives you cleaner type hints
import torch
import numpy as np 
from pathlib import Path
from typing import List, Union
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers import Tokenizer
import os, json

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
    
    
class BPETokenizer : 
    """This is essentially a huggingface tokenizer wrapper -> Fast implementation in RUST"""
    
    def __init__(self, vocab_size : int = 32000, special_tokens : list[str] = None) : 
        if ByteLevelBPETokenizer is None : 
            raise ImportError("Please `pip install tokenizers` for BPETokenizers")
        
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None
        
    def train(self, data_path : Union[str, Path]) : 
        files : List[str] = []
        p = Path(data_path)
        
        if p.is_dir() : 
            files = [str(fp) for fp in p.glob("**/*.txt")]
        else : 
            files = [str(p)]
            
        tok = ByteLevelBPETokenizer()
        # There is a method called a train from iterator, that is going to be efficient implementation
        tok.train(
            files=files,
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens
            )
        self._tok = tok
        
    def save(self, out_dir : Union[str, Path]) : 
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self._tok.save_model(directory=str(out))
        self._tok.save(path=str(out / "tokenizer.json"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        (out/"bpe_meta.json").write_text(json.dumps(meta))
        
    def load(self, dir_path : Union[str, Path]) : 
        dirp = Path(dir_path)
        vocab = dirp / "vocab.json"
        merges = dirp / "merges.txt"
        tokenizer = dirp / "tokenizer.json"
        
        # Fallback case 
        if not vocab.exists() or not merges.exists() : 
            vs = list(dirp.glob("*.json"))
            ms = list(dirp.glob("*.txt"))
            if not vs or not ms:
                raise FileNotFoundError(f"Could not find vocab.json/merges.txt in {dirp}")
            vocab = vs[0]
            merges = ms[0]
            
        tok = Tokenizer.from_file(str(tokenizer))
        self._tok = tok
        meta_file = dirp / "bpe_meta.json"
        
        if meta_file.exists() : 
            meta = json.loads(meta_file.read_text())
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)
            
    def encode(self, text : str) : 
        ids = self._tok.encode(text).ids
        return ids
    
    def decode(self, ids) : 
        return self._tok.decode(ids)
  
  
    
class SentencePieceTokenizer : 
    """TODO : Implment this later"""
    
    