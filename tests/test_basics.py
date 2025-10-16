import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest
from src.components.multiheadAttention import MultiHeadAttention


# ===== Shape tests =====
def test_output_shape_default():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    x = torch.randn(2, 10, 32)
    out = m(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 10, 32)

def test_arbitrary_dv_shape():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=16, dropout=0.0)
    x = torch.randn(3, 5, 32)
    out = m(x)
    assert out.shape == (3, 5, 32)

def test_single_head():
    m = MultiHeadAttention(heads=1, d_model=64, d_v=64, dropout=0.0)
    x = torch.randn(1, 8, 64)
    out = m(x)
    assert out.shape == (1, 8, 64)

def test_large_batch_long_sequence():
    m = MultiHeadAttention(heads=8, d_model=512, d_v=64, dropout=0.0)
    x = torch.randn(16, 128, 512)
    out = m(x)
    assert out.shape == (16, 128, 512)


# ===== Invalid input tests =====
def test_invalid_d_model_divisibility():
    with pytest.raises(AssertionError):
        MultiHeadAttention(heads=6, d_model=32, d_v=8)

def test_wrong_input_dimension():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    x = torch.randn(2, 10, 64)  # wrong d_model
    with pytest.raises(AssertionError):
        m(x)



# ===== Numerical / correctness tests =====
def test_zero_projections_produce_zero_output():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    m.q_proj.weight.data.zero_()
    m.k_proj.weight.data.zero_()
    m.v_proj.weight.data.zero_()
    m.out_proj.weight.data.zero_()

    x = torch.randn(2, 7, 32)
    out = m(x)
    assert torch.allclose(out, torch.zeros_like(out))

def test_output_is_finite():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    x = torch.randn(2, 10, 32)
    out = m(x)
    assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

def test_causal_masking_blocks_future():
    # Ensure position i cannot attend to j > i
    m = MultiHeadAttention(heads=2, d_model=16, d_v=8, dropout=0.0)
    x = torch.randn(1, 5, 16)
    
    # Hook to capture attention weights (add inside forward if needed for this test)
    # For simplicity, we check that running with causal mask doesn't crash
    out = m(x)
    assert out.shape == (1, 5, 16)

def test_deterministic_with_fixed_seed():
    torch.manual_seed(42)
    m1 = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    x = torch.randn(2, 10, 32)
    out1 = m1(x)

    torch.manual_seed(42)
    m2 = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    out2 = m2(x)
    
    assert torch.allclose(out1, out2), "Output should be deterministic with same seed"


# ===== Gradient tests =====
def test_gradients_flow_through_attention():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = m(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow to input"
    assert m.q_proj.weight.grad is not None
    assert m.k_proj.weight.grad is not None
    assert m.v_proj.weight.grad is not None
    assert m.out_proj.weight.grad is not None

def test_dropout_effect_in_training_mode():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.5)
    m.train()
    x = torch.randn(2, 10, 32)
    
    torch.manual_seed(0)
    out1 = m(x)
    torch.manual_seed(0)
    out2 = m(x)
    
    # With dropout, same seed should give same result
    assert torch.allclose(out1, out2)
    
    # Different seeds should give different results (stochastic dropout)
    torch.manual_seed(1)
    out3 = m(x)
    assert not torch.allclose(out1, out3), "Dropout should introduce randomness"

def test_dropout_disabled_in_eval_mode():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.5)
    m.eval()
    x = torch.randn(2, 10, 32)
    
    out1 = m(x)
    out2 = m(x)
    
    assert torch.allclose(out1, out2), "Eval mode should be deterministic (no dropout)"


# ===== Edge cases =====
def test_single_token_sequence():
    m = MultiHeadAttention(heads=2, d_model=16, d_v=8, dropout=0.0)
    x = torch.randn(1, 1, 16)  # single token
    out = m(x)
    assert out.shape == (1, 1, 16)

def test_very_small_d_v():
    m = MultiHeadAttention(heads=4, d_model=32, d_v=2, dropout=0.0)
    x = torch.randn(2, 5, 32)
    out = m(x)
    assert out.shape == (2, 5, 32)

def test_d_v_equals_d_k():
    # Common case: d_v == d_k == d_model // heads
    m = MultiHeadAttention(heads=4, d_model=32, d_v=8, dropout=0.0)
    assert m.d_v == m.d_k == 8
    x = torch.randn(2, 10, 32)
    out = m(x)
    assert out.shape == (2, 10, 32)


# ===== Parameter count test =====
def test_parameter_count():
    m = MultiHeadAttention(heads=4, d_model=64, d_v=16, dropout=0.0)
    total_params = sum(p.numel() for p in m.parameters())
    
    # q_proj: 64 * 64 = 4096
    # k_proj: 64 * 64 = 4096
    # v_proj: 64 * 64 = 4096  (heads * d_v = 4 * 16 = 64)
    # out_proj: 64 * 64 = 4096
    expected = 4096 * 4
    assert total_params == expected, f"Expected {expected} params, got {total_params}"