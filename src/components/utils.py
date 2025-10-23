"""We are going to filter by the sum of top p probability or top k"""

from __future__ import annotations
import torch

def top_k_top_p_filtering(logits : torch.Tensor, top_k : int, top_p : float) : 
    B, V = logits.shape # BATCH, VOCAB
    filtered = logits.clone()
    
    if top_k is not None and top_k < V : 
        topk_vals, _ = torch.topk(filtered, top_k, dim = -1) # Filtering through the Vocab
        kth = topk_vals[:, -1].unsqueeze(-1) 
        filtered[filtered < kth] = float("-inf") # Setting the values
        
    if top_p is not None and 0 < top_p < 1 : 
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim = -1)
        cumsum = torch.cumsum(probs, dim =-1) # Calculating the cummulative sum
        mask = cumsum > top_p
        mask[..., 0] = False
        sorted_logits[mask] = float("-inf")
        filtered = torch.full_like(filtered, float("-inf")) # Fills every element with -inf
        filtered.scatter_(1, sorted_idx, sorted_logits) # Scatters at position specified.
        
    return filtered


        
    
    