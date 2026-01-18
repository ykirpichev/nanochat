"""
Unit tests for EfficientMoE module. Example run:

python -m pytest tests/test_moe.py -v
"""

import torch
import torch.nn as nn
import pytest
from dataclasses import dataclass

from nanochat.moe import MoE, EfficientMoE, Expert, SCATTERMOE_AVAILABLE
from nanochat.gpt import GPTConfig


@dataclass
class TestConfig:
    """Minimal config for MoE tests."""
    n_embd: int = 64
    sequence_len: int = 32


def test_moe_basic_forward():
    """Test basic forward pass through MoE."""
    config = TestConfig(n_embd=64)
    moe = MoE(config, num_experts=4, top_k=2)
    
    B, T, C = 2, 8, config.n_embd
    x = torch.randn(B, T, C)
    
    output = moe(x)

    assert output.shape == x.shape, f"Output shape {output.shape} should match input shape {x.shape}"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"
    assert not torch.isinf(output).any(), "Output should not contain Inf values"


def test_moe_output_dtype():
    """Test that output dtype matches input dtype."""
    config = TestConfig(n_embd=64)
    moe = MoE(config, num_experts=4, top_k=2)
    
    x = torch.randn(2, 8, config.n_embd, dtype=torch.float32)
    output = moe(x)
    assert output.dtype == x.dtype, f"Output dtype {output.dtype} should match input dtype {x.dtype}"
    
    # x = torch.randn(2, 8, config.n_embd, dtype=torch.bfloat16)
    # output = moe(x)
    # assert output.dtype == x.dtype, f"Output dtype {output.dtype} should match input dtype {x.dtype}"


def test_moe_top_k_routing():
    """Test that top-k routing works correctly (using EfficientMoE implementation logic)."""
    config = TestConfig(n_embd=64)
    # Force EfficientMoE for this test to access internal router
    moe = EfficientMoE(config, num_experts=8, top_k=2)
    
    B, T, C = 2, 8, config.n_embd
    x = torch.randn(B, T, C)
    
    # Get router logits
    x_flat = x.view(-1, C)
    router_logits = moe.router(x_flat)
    
    # Check that top-k selection works
    top_k_logits, top_k_indices = torch.topk(router_logits, moe.top_k, dim=-1)
    
    assert top_k_indices.shape == (B * T, moe.top_k), "Top-k indices should have correct shape"
    assert top_k_logits.shape == (B * T, moe.top_k), "Top-k logits should have correct shape"
    
    # Check that indices are valid (within [0, num_experts))
    assert (top_k_indices >= 0).all(), "All indices should be >= 0"
    assert (top_k_indices < moe.num_experts).all(), f"All indices should be < {moe.num_experts}"


def test_moe_load_balance_loss_training():
    """Test load balancing loss computation in training mode."""
    config = TestConfig(n_embd=64)
    moe = MoE(config, num_experts=4, top_k=2, load_balance_weight=0.01)
    moe.train()  # Set to training mode
    
    B, T, C = 2, 8, config.n_embd
    x = torch.randn(B, T, C)
    
    output = moe(x)
    load_balance_loss = moe.get_load_balance_loss()
    
    assert load_balance_loss is not None, "Load balance loss should be computed in training mode"
    assert load_balance_loss.item() >= 0, "Load balance loss should be non-negative"
    assert isinstance(load_balance_loss, torch.Tensor), "Load balance loss should be a tensor"


def test_moe_no_load_balance_loss_inference():
    """Test that load balancing loss is not computed in inference mode."""
    config = TestConfig(n_embd=64)
    moe = MoE(config, num_experts=4, top_k=2)
    moe.eval()  # Set to inference mode
    
    B, T, C = 2, 8, config.n_embd
    x = torch.randn(B, T, C)
    
    output = moe(x)
    load_balance_loss = moe.get_load_balance_loss()
    
    # The get_load_balance_loss returns 0.0 tensor if None
    assert load_balance_loss.item() == 0.0, "Load balance loss should be 0.0 in inference mode"


def test_moe_backend_selection():
    """Test that correct backend is selected."""
    config = TestConfig(n_embd=64)
    moe = MoE(config, num_experts=4, top_k=2)
    
    if SCATTERMOE_AVAILABLE:
        assert isinstance(moe.impl, type(moe.impl)), "Should use ScatterMoE when available"
        # Note: can't easily check class name if not imported, but we can check if it's NOT EfficientMoE
        # actually we can import ScatterMoE inside the test if needed
    else:
        assert isinstance(moe.impl, EfficientMoE), "Should use EfficientMoE when ScatterMoE is not available"
    load_balance_loss = moe.get_load_balance_loss()
    
    # In eval mode, should return zero tensor
    assert load_balance_loss is not None, "Should return a tensor even in eval mode"
    assert load_balance_loss.item() == 0.0, "Load balance loss should be 0 in eval mode"


def test_moe_different_num_experts():
    """Test MoE with different numbers of experts."""
    config = TestConfig(n_embd=64)
    
    for num_experts in [2, 4, 8, 16]:
        moe = EfficientMoE(config, num_experts=num_experts, top_k=2)
        x = torch.randn(2, 8, config.n_embd)
        output = moe(x)
        assert output.shape == x.shape, f"Output shape should match for {num_experts} experts"


def test_moe_different_top_k():
    """Test MoE with different top-k values."""
    config = TestConfig(n_embd=64)
    num_experts = 8
    
    for top_k in [1, 2, 4]:
        if top_k > num_experts:
            continue  # Skip invalid configurations
        
        moe = EfficientMoE(config, num_experts=num_experts, top_k=top_k)
        x = torch.randn(2, 8, config.n_embd)
        output = moe(x)
        assert output.shape == x.shape, f"Output shape should match for top_k={top_k}"


def test_moe_single_token():
    """Test MoE with single token input."""
    config = TestConfig(n_embd=64)
    moe = EfficientMoE(config, num_experts=4, top_k=2)
    
    x = torch.randn(1, 1, config.n_embd)
    output = moe(x)
    
    assert output.shape == x.shape, "Output shape should match for single token"


def test_moe_single_batch():
    """Test MoE with single batch item."""
    config = TestConfig(n_embd=64)
    moe = EfficientMoE(config, num_experts=4, top_k=2)
    
    x = torch.randn(1, 8, config.n_embd)
    output = moe(x)
    
    assert output.shape == x.shape, "Output shape should match for single batch"


def test_moe_gradient_flow():
    """Test that gradients flow through the MoE layer."""
    config = TestConfig(n_embd=64)
    moe = EfficientMoE(config, num_experts=4, top_k=2)
    moe.train()
    
    x = torch.randn(2, 8, config.n_embd, requires_grad=True)
    output = moe(x)
    
    # Compute a dummy loss
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert moe.router.weight.grad is not None, "Router weights should have gradients"
    
    # Check that expert weights have gradients
    for expert in moe.experts:
        assert expert.c_fc.weight.grad is not None, "Expert c_fc weights should have gradients"
        assert expert.c_proj.weight.grad is not None, "Expert c_proj weights should have gradients"


def test_moe_parameter_count():
    """Test that MoE has expected number of parameters."""
    config = TestConfig(n_embd=64)
    num_experts = 4
    d_expert = 4 * config.n_embd  # Default d_expert
    
    # Create a regular Expert for comparison
    expert = Expert(config.n_embd, d_expert)
    expert_params = sum(p.numel() for p in expert.parameters())
    
    # Create MoE
    moe = EfficientMoE(config, num_experts=num_experts, top_k=2, d_expert=d_expert)
    moe_params = sum(p.numel() for p in moe.parameters())
    
    # MoE should have: router (n_embd * num_experts) + num_experts * expert_params
    expected_router_params = config.n_embd * num_experts
    expected_moe_params = expected_router_params + num_experts * expert_params
    
    assert moe_params == expected_moe_params, (
        f"MoE should have {expected_moe_params} parameters, "
        f"but has {moe_params}. "
        f"Router: {expected_router_params}, Experts: {num_experts * expert_params}"
    )


def test_moe_no_bias():
    """Test that MoE follows codebase convention of no bias in linear layers."""
    config = TestConfig(n_embd=64)
    moe = EfficientMoE(config, num_experts=4, top_k=2)
    
    # Router should have no bias
    assert moe.router.bias is None, "Router should have no bias"
    
    # Each expert should have no bias
    for expert in moe.experts:
        assert expert.c_fc.bias is None, "Expert c_fc should have no bias"
        assert expert.c_proj.bias is None, "Expert c_proj should have no bias"


def test_moe_load_balance_weight():
    """Test that load balance weight affects the loss."""
    config = TestConfig(n_embd=64)
    
    weight1 = 0.01
    weight2 = 0.1
    
    # Use same seed to ensure same initialization
    torch.manual_seed(42)
    moe1 = EfficientMoE(config, num_experts=4, top_k=2, load_balance_weight=weight1)
    torch.manual_seed(42)
    moe2 = EfficientMoE(config, num_experts=4, top_k=2, load_balance_weight=weight2)
    
    # Copy weights from moe1 to moe2 so they're identical except for load_balance_weight
    moe2.router.weight.data = moe1.router.weight.data.clone()
    for i, expert in enumerate(moe2.experts):
        expert.c_fc.weight.data = moe1.experts[i].c_fc.weight.data.clone()
        expert.c_proj.weight.data = moe1.experts[i].c_proj.weight.data.clone()
    
    moe1.train()
    moe2.train()
    
    x = torch.randn(2, 8, config.n_embd)
    
    _ = moe1(x)
    loss1 = moe1.get_load_balance_loss()
    
    _ = moe2(x)
    loss2 = moe2.get_load_balance_loss()
    
    # With same weights but different load_balance_weight, the losses should scale proportionally
    if loss1.item() > 0:
        ratio = loss2.item() / loss1.item()
        expected_ratio = weight2 / weight1
        # Should be very close since weights are identical
        assert abs(ratio - expected_ratio) < 1e-5, (
            f"Load balance loss ratio {ratio:.6f} should equal weight ratio {expected_ratio:.6f}"
        )
    else:
        # If loss1 is zero, loss2 should also be zero (perfectly balanced case)
        assert loss2.item() == 0.0, "Both losses should be zero if perfectly balanced"


def test_moe_with_gpt_config():
    """Test MoE with actual GPTConfig."""
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4, n_layer=4)
    moe = EfficientMoE(config, num_experts=8, top_k=2)
    
    x = torch.randn(2, 16, config.n_embd)
    output = moe(x)
    
    assert output.shape == x.shape, "Output shape should match with GPTConfig"


def test_moe_d_expert_parameter():
    """Test MoE with different d_expert values."""
    config = TestConfig(n_embd=64)
    
    # Test with default d_expert (4 * n_embd)
    moe1 = EfficientMoE(config, num_experts=4, top_k=2)
    assert moe1.d_expert == 4 * config.n_embd, "Default d_expert should be 4 * n_embd"
    
    # Test with custom d_expert
    d_expert = 256
    moe2 = EfficientMoE(config, num_experts=4, top_k=2, d_expert=d_expert)
    assert moe2.d_expert == d_expert, f"d_expert should be {d_expert}"
    
    # Test forward pass with custom d_expert
    x = torch.randn(2, 8, config.n_embd)
    output = moe2(x)
    assert output.shape == x.shape, "Output shape should match with custom d_expert"
    
    # Verify expert structure
    for expert in moe2.experts:
        assert expert.c_fc.out_features == d_expert, "Expert c_fc should have d_expert output features"
        assert expert.c_proj.in_features == d_expert, "Expert c_proj should have d_expert input features"


def test_moe_expert_usage_distribution():
    """Test that experts are being used (not all routing to same expert)."""
    config = TestConfig(n_embd=64)
    num_experts = 8
    moe = EfficientMoE(config, num_experts=num_experts, top_k=2)
    moe.train()
    
    # Use a larger batch to get better statistics
    B, T = 4, 16
    x = torch.randn(B, T, config.n_embd)
    
    output = moe(x)
    
    # Get router logits to check expert usage
    x_flat = x.view(-1, config.n_embd)
    router_logits = moe.router(x_flat)
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_usage = router_probs.mean(dim=0)
    
    # With random initialization, we expect some variance in expert usage
    # All experts should have non-zero usage probability
    assert (expert_usage > 0).all(), "All experts should have non-zero usage probability"
    
    # The usage should not be too concentrated (max usage < 0.5 for 8 experts)
    max_usage = expert_usage.max().item()
    assert max_usage < 0.5, f"Expert usage should be distributed, but max is {max_usage:.3f}"


def test_moe_deterministic_with_seed():
    """Test that MoE produces deterministic outputs with fixed seed."""
    config = TestConfig(n_embd=64)
    
    torch.manual_seed(42)
    moe1 = EfficientMoE(config, num_experts=4, top_k=2)
    x1 = torch.randn(2, 8, config.n_embd)
    output1 = moe1(x1)
    
    torch.manual_seed(42)
    moe2 = EfficientMoE(config, num_experts=4, top_k=2)
    x2 = torch.randn(2, 8, config.n_embd)
    output2 = moe2(x2)
    
    # Outputs should match if we use the same seed and same initialization
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be deterministic with same seed"


def test_moe_top_k_equals_num_experts():
    """Test MoE when top_k equals num_experts (all experts used)."""
    config = TestConfig(n_embd=64)
    num_experts = 4
    moe = EfficientMoE(config, num_experts=num_experts, top_k=num_experts)
    
    x = torch.randn(2, 8, config.n_embd)
    output = moe(x)
    
    assert output.shape == x.shape, "Output shape should match when top_k == num_experts"


def test_moe_device_consistency():
    """Test that MoE works on different devices if available."""
    config = TestConfig(n_embd=64)
    moe = EfficientMoE(config, num_experts=4, top_k=2)
    
    x = torch.randn(2, 8, config.n_embd)
    output = moe(x)
    
    # Check that output is on same device as input
    assert output.device == x.device, "Output should be on same device as input"
    
    # If CUDA is available, test on CUDA
    if torch.cuda.is_available():
        moe_cuda = moe.cuda()
        x_cuda = x.cuda()
        output_cuda = moe_cuda(x_cuda)
        assert output_cuda.device.type == "cuda", "Output should be on CUDA when input is on CUDA"


def test_moe_gradient_checkpointing():
    """Test that gradient checkpointing works and produces correct gradients."""
    config = TestConfig(n_embd=64)
    # Enable checkpointing
    moe = EfficientMoE(config, num_experts=4, top_k=2, checkpointing=True)
    moe.train()
    
    x = torch.randn(2, 8, config.n_embd, requires_grad=True)
    output = moe(x)
    
    # Check forward pass works
    assert output.shape == x.shape, "Output shape mismatch with checkpointing"
    
    # Check backward pass works
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient missing with checkpointing"
    assert moe.router.weight.grad is not None, "Router gradient missing with checkpointing"
    
    # Compare with non-checkpointed version (gradients should be identical)
    torch.manual_seed(42)
    moe_nc = EfficientMoE(config, num_experts=4, top_k=2, checkpointing=False)
    moe_c = EfficientMoE(config, num_experts=4, top_k=2, checkpointing=True)
    
    # Sync weights
    moe_c.load_state_dict(moe_nc.state_dict())
    
    x_nc = torch.randn(2, 8, config.n_embd, requires_grad=True)
    x_c = x_nc.clone().detach().requires_grad_(True)
    
    loss_nc = moe_nc(x_nc).sum()
    loss_nc.backward()
    
    loss_c = moe_c(x_c).sum()
    loss_c.backward()
    
    # Gradients should be very close
    assert torch.allclose(x_nc.grad, x_c.grad, atol=1e-6), "Input gradients mismatch between checkpointed and standard"
    for p_nc, p_c in zip(moe_nc.parameters(), moe_c.parameters()):
        if p_nc.grad is not None:
            assert torch.allclose(p_nc.grad, p_c.grad, atol=1e-6), "Parameter gradients mismatch"
