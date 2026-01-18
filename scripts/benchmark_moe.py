
import time
import torch
import torch.nn as nn
import sys
import os

# Add project root to sys.path to allow running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanochat.moe import MoE, EfficientMoE, Expert
from nanochat.gpt import GPTConfig, MLP

def benchmark_module(module, x, name="Module", steps=100, warmup=10):
    module.eval()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x)
            
    # Benchmark
    start_time = time.time()
    for _ in range(steps):
        with torch.no_grad():
            _ = module(x)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / steps
    
    num_params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {avg_time*1000:.2f} ms per step | Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    return avg_time

def benchmark_training(module, x, name="Module", steps=100, warmup=10):
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        output = module(x)
        loss = output.mean()
        # Add load balancing loss if available
        if hasattr(module, 'get_load_balance_loss'):
            loss += module.get_load_balance_loss()
        loss.backward()
        optimizer.step()
            
    # Benchmark
    start_time = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        output = module(x)
        loss = output.mean()
        # Add load balancing loss if available
        if hasattr(module, 'get_load_balance_loss'):
            loss += module.get_load_balance_loss()
        loss.backward()
        optimizer.step()
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / steps
    
    num_params = sum(p.numel() for p in module.parameters())
    print(f"{name} (Train): {avg_time*1000:.2f} ms per step | Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    return avg_time

def main():
    print("Benchmarking MoE vs MLP (Inference)...")
    
    config = GPTConfig(n_embd=768)
    B, T = 8, 1024
    x = torch.randn(B, T, config.n_embd)
    
    print(f"Input shape: {x.shape}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if torch.cuda.is_available():
        x = x.cuda()
        
    # Standard MLP
    mlp = MLP(config)
    if torch.cuda.is_available():
        mlp = mlp.cuda()
    benchmark_module(mlp, x, name="Standard MLP")
    
    # MoE (8 experts, top-2)
    moe = MoE(config, num_experts=8, top_k=2)
    if torch.cuda.is_available():
        moe = moe.cuda()
    benchmark_module(moe, x, name="MoE (8 experts, top-2)")
    
    # MoE (16 experts, top-2)
    moe16 = MoE(config, num_experts=16, top_k=2)
    if torch.cuda.is_available():
        moe16 = moe16.cuda()
    benchmark_module(moe16, x, name="MoE (16 experts, top-2)")
    
    print("\nBenchmarking MoE vs MLP (Training)...")
    
    # Re-initialize models for training benchmark to start fresh
    mlp = MLP(config)
    if torch.cuda.is_available(): mlp = mlp.cuda()
    benchmark_training(mlp, x, name="Standard MLP")
    
    moe = MoE(config, num_experts=8, top_k=2)
    if torch.cuda.is_available(): moe = moe.cuda()
    benchmark_training(moe, x, name="MoE (8 experts, top-2)")
    
    moe16 = MoE(config, num_experts=16, top_k=2)
    if torch.cuda.is_available(): moe16 = moe16.cuda()
    benchmark_training(moe16, x, name="MoE (16 experts, top-2)")

if __name__ == "__main__":
    main()
