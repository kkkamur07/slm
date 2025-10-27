# SLM: Small Language Model

A PyTorch implementation of a Small Language Model with state-of-the-art optimizations.

## Features

- **Multi-Head Attention**: Scaled dot-product attention with causal masking
- **Group Query Attention (GQA)**: Reduced memory footprint
- **KV Caching**: Fast inference without recomputation
- **Different Tokenizations**: Implementing sentence piece, byte pair encoding and other. 
- **RLHF & DPO**: Advanced training techniques

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
uv sync
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

## License

MIT