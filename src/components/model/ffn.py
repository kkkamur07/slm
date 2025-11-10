import torch.nn as nn

class FeedForwardNetwork(nn.Module) : 
    def __init__(self, d_model : int, mul_factor : int, dropout : float = 0.0) : 
        super().__init__()
        self.d_model = d_model
        self.mul_factor = mul_factor
        self.dropout = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            nn.Linear(d_model, mul_factor * d_model),
            nn.GELU(),
            nn.Linear(d_model * mul_factor, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, input) : 
        return self.net(input)