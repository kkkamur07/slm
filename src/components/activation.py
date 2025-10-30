import torch.nn as nn

class SwiGLU(nn.Module) : 
    def __init__(self, dim : int, mult:int = 4, dropout : float = 0.0) : 
        super().__init__()
        inner = mult * dim # inner dimension
        self.w1 = nn.Linear(dim, inner, bias = False)
        self.w2 = nn.Linear(dim, inner, bias = False)
        self.w3 = nn.Linear(inner, dim, bias = False)
        
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self,x) : 
        a = self.w1(x) 
        b = self.act(self.w2(x))
        
        return self.drop(self.w3(a*b)) # We are just adding the transformed version. 
    
    