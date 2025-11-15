import torch
from src.components.model.rms_norm import RMSNorm

def test_rmsnorm_shapes():
    x = torch.randn(2,3,8)
    rn = RMSNorm(8)
    y = rn(x)
    assert y.shape == x.shape