import torch 
import torch.nn as nn
from .block import Block
from ..utils import top_k_top_p_filtering
import torch.nn.functional as F

class GPT(nn.Module) : 
    def __init__(self, vocab_size, block_size, n_layer, heads, d_model, dropout ) : 
        super().__init__()
        self.block_size = block_size # Maximum number of tokens per sequence
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_layer = n_layer
        self.heads = heads
        
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.block_size, self.d_model) 
        
        self.blocks = nn.ModuleList([
            Block(d_model=self.d_model, 
                  heads=self.heads, 
                  dropout=self.dropout) 
            for _ in range(self.n_layer)])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False) # INPUT : d_model OUTPUT : vocab_size
        
        self.apply(self._init_weights)
    
    # HE initialization
    def _init_weights(self,m) : 
        if isinstance(m, nn.Linear) : 
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None : 
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
    # Taking in input and gives you output
    def forward(self, idx : torch.Tensor, targets : torch.Tensor | None) : 
        batch, tokens = idx.shape
        assert tokens <= self.block_size 
        pos = torch.arange(0,tokens, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos) 
        x = self.dropout(x)
        
        for blocks in self.blocks : 
            x = blocks(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        
        if targets is not None : 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad() # Defining the property here
    def generate(self, idx : torch.Tensor, max_new_tokens : int = 200, temperature : float = 1.0, top_k : int | None = 50, top_p : float | None = None)  :
        self.eval()
        
        if idx.size(1) == 0 : 
            idx = torch.full((idx.size(0), 1),10, dtype=torch.long, device=idx.device)
        
        for _ in range(max_new_tokens) : 
            idx_cond = idx[:, -self.block_size:] # We only need generation till block_size
            logits, _ = self(idx_cond) # Calls the forward method. 
            logits = logits[:, -1, :] / max(temperature, 1e-6) # Temperature is never zero
            logits = top_k_top_p_filtering(logits=logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim = -1)
            next_id = torch.multinomial(probs, num_samples=1) # Sampling, people are experimenting with this.
            idx = torch.cat([idx, next_id], dim = 1) # Extending
        return idx
    
    
        