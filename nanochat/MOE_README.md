# Sparse Mixture of Experts (MoE) for NanoChat

This module provides a production-ready implementation of Sparse Mixture of Experts (MoE) for the NanoChat architecture. It is designed to be efficient on both CPU (via vectorized PyTorch operations) and GPU (via [ScatterMoE](https://github.com/shawntan/scattermoe) integration).

## Features

- **Top-K Routing**: Configurable sparse routing (typically k=1 or k=2) where only a subset of experts process each token.
- **Load Balancing**: Auxiliary loss function to ensure experts are utilized evenly, preventing "expert collapse".
- **Dual Backend**:
  - **EfficientMoE**: A highly optimized pure PyTorch implementation for CPU/MPS.
  - **ScatterMoE**: Seamless integration with Triton-based ScatterMoE for high-performance GPU training and inference.
- **Unified Interface**: A single `MoE` class that automatically selects the best available backend.

## Installation

The core functionality works with standard PyTorch. For GPU acceleration, you need `scattermoe` and `triton`.

```bash
# Standard installation
pip install .

# For GPU acceleration (requires CUDA)
pip install git+https://github.com/shawntan/scattermoe.git
```

## Usage

### Basic Integration

Replace standard MLP layers with the `MoE` module in your transformer blocks.

```python
from nanochat.moe import MoE

# Configuration
class Config:
    n_embd = 768

config = Config()

# Initialize MoE layer
# num_experts: Total number of experts
# top_k: Number of experts active per token
# load_balance_weight: Weight for the auxiliary load balancing loss
moe_layer = MoE(
    config, 
    num_experts=8, 
    top_k=2, 
    load_balance_weight=0.01
)

# Forward pass
output = moe_layer(x)

# Training loop integration
# You must add the load balancing loss to your main objective
loss = criterion(output, target) + moe_layer.get_load_balance_loss()
loss.backward()
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 8 | Total number of experts in the mixture. |
| `top_k` | 2 | Number of experts selected for each token. |
| `load_balance_weight` | 0.01 | Coefficient for the load balancing loss. |
| `d_expert` | 4 * n_embd | Hidden dimension size of each expert MLP. |

## Architecture Details

### Routing Mechanism
We use a learned router (linear layer) to predict the best experts for each token. The top-k experts with the highest logits are selected. The output is a weighted sum of the selected experts' outputs, weighted by the softmax probability of the router logits.

### Load Balancing
To ensure efficient training, we add an auxiliary loss that penalizes the variance in expert usage. This encourages the router to distribute tokens uniformly across all experts. The loss is calculated as the squared coefficient of variation of the expert assignment probabilities.

### Backends

1.  **EfficientMoE (CPU/Default)**:
    - Uses vectorized operations to process experts.
    - Optimized loops to avoid excessive Python overhead.
    - Suitable for debugging, CPU inference, and environments without Triton support.

2.  **ScatterMoE (GPU)**:
    - Enabled automatically if `scattermoe` is installed.
    - Uses custom Triton kernels for high-throughput scattering and gathering of tokens.
    - Supports kernel fusion for maximum efficiency on NVIDIA GPUs.

## Benchmarking

A benchmark script is provided in `scripts/benchmark_moe.py` to compare MoE performance against standard MLPs.

```bash
python scripts/benchmark_moe.py
```

## Testing

Run the unit tests to verify correctness:

```bash
python -m pytest tests/test_moe.py
```
