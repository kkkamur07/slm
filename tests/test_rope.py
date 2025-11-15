import torch
from src.components.pos import ROPE

# Vanilla case: same #heads for q and k
def test_rope_rotation_shapes_single():
    B, H, T, D = 1, 2, 5, 8
    
    rope = ROPE(head_dim=D, max_seq_len=32) 
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    
    q2 = rope(q, seq_len=T, start_pos=0)
    k2 = rope(k, seq_len=T, start_pos=0)
    
    assert q2.shape == q.shape
    assert k2.shape == k.shape

    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)

# GQA case: q has H heads; k has fewer Hk heads (shared KV)
def test_rope_rotation_shapes_gqa():
    B, H, Hk, T, D = 2, 8, 2, 7, 16

    rope = ROPE(head_dim=D, max_seq_len=128)  
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, Hk, T, D)
    
    q2 = rope(q, seq_len=T, start_pos=10)
    k2 = rope(k, seq_len=T, start_pos=10)
    
    assert q2.shape == (B, H, T, D)
    assert k2.shape == (B, Hk, T, D)
    
    # Rotations should be deterministic for same positions
    # Check that values changed
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)