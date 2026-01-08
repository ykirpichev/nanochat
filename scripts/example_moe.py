
import torch
from nanochat.moe import MoE
from nanochat.gpt import GPTConfig

def main():
    # 1. Configuration
    config = GPTConfig(n_embd=768)
    num_experts = 8
    top_k = 2
    
    print(f"Initializing MoE with {num_experts} experts, top-{top_k} routing...")
    
    # 2. Initialize MoE module
    # Automatically uses ScatterMoE (Triton) if available and on GPU, otherwise EfficientMoE (PyTorch)
    moe = MoE(config, num_experts=num_experts, top_k=top_k)
    
    # 3. Create dummy input
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    # 4. Forward pass (Inference)
    moe.eval()
    with torch.no_grad():
        output = moe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 5. Training loop example (Load Balancing)
    moe.train()
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-4)
    
    print("\nTraining step example...")
    output = moe(x)
    
    # Main task loss (dummy)
    task_loss = output.mean()
    
    # Load balancing loss
    lb_loss = moe.get_load_balance_loss()
    
    total_loss = task_loss + lb_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Task Loss: {task_loss.item():.4f}")
    print(f"Load Balance Loss: {lb_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    
    print("\nSuccess! MoE module is ready for use.")

if __name__ == "__main__":
    main()
