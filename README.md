# SLM: Small Language Model

A PyTorch implementation of a Small Language Model with state-of-the-art optimizations.

## Features

- **Multi-Head Attention**: Scaled dot-product attention with causal masking
- **Group Query Attention (GQA)**: Reduced memory footprint
- **KV Caching**: Fast inference without recomputation
- **RLHF & DPO**: Advanced training techniques
- **Comprehensive Tests**: 18+ unit tests included

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pytest
```

## Quick Start

```python
from src.basics.multiheadAttention import MultiHeadAttention
import torch

mha = MultiHeadAttention(heads=8, d_model=512, d_v=64)
x = torch.randn(2, 10, 512)
output = mha(x)  # (2, 10, 512)
```

## Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

## Project Structure

```
src/
├── basics/
│   ├── multiheadAttention.py
│   └── mask.py
├── model/
│   └── slm.py
└── training/
    ├── sft.py
    ├── rlhf.py
    └── dpo.py
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Flash-Attention](https://arxiv.org/abs/2205.14135)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2306.04604)

## License

MIT