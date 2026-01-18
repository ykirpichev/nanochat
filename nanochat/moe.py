"""
Efficient Mixture of Experts (MoE) module for sparse expert routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from scattermoe.mlp import MLP as ScatterMoE_MLP
    SCATTERMOE_AVAILABLE = True
except ImportError:
    SCATTERMOE_AVAILABLE = False
    ScatterMoE_MLP = None


class Expert(nn.Module):
    """
    Expert MLP module for MoE.
    Follows codebase conventions: no bias, relu^2 activation.
    """
    def __init__(self, n_embd, d_expert):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, d_expert, bias=False)
        self.c_proj = nn.Linear(d_expert, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class CPUEfficientMoE(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = config.n_embd
        self.d_ff = 1 * config.n_embd

        self.router = nn.Linear(self.d_model, num_experts, bias=False)
        
        # Consolidate all experts into one big parameter tensor
        # Shape: (num_experts, input_dim, output_dim)
        self.w1 = nn.Parameter(torch.randn(num_experts, self.d_model, self.d_ff))
        self.w2 = nn.Parameter(torch.randn(num_experts, self.d_ff, self.d_model))

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C) # (N, C)
        
        # 1. Routing
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1) # (N, k)
        
        # 2. Vectorized Expert Processing (No Python Loop!)
        # We project the input for ALL experts simultaneously
        # (N, C) -> (N, k, C)
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1)
        
        # Select the weights for each token's top-k experts
        # Using advanced indexing to gather weights: (N, k, C, D_ff)
        w1_selected = self.w1[indices] 
        w2_selected = self.w2[indices]
        
        # First Layer: (N, k, 1, C) @ (N, k, C, D_ff) -> (N, k, 1, D_ff)
        hidden = torch.matmul(x_expanded.unsqueeze(2), w1_selected)
        hidden = F.relu(hidden)
        
        # Second Layer: (N, k, 1, D_ff) @ (N, k, D_ff, C) -> (N, k, 1, C)
        out = torch.matmul(hidden, w2_selected).squeeze(2)
        
        # 3. Apply routing weights and sum
        out = (out * weights.unsqueeze(-1)).sum(dim=1)
        
        return out.view(B, T, C)

class CPUMaxEfficiencyMoE(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = config.n_embd
        self.d_ff = 1 * config.n_embd

        self.router = nn.Linear(self.d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_ff, bias=False),
                nn.ReLU(), # We will square this manually for speed
                nn.Linear(self.d_ff, self.d_model, bias=False)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # 1. Routing
        logits = self.router(x_flat)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        
        # 2. Pre-allocate output
        final_output = torch.zeros_like(x_flat)
        
        # 3. Process by "Flattened Top-K"
        # We treat each (token, k) pair as a separate task
        flat_indices = indices.view(-1)      # (N*k)
        flat_weights = weights.view(-1, 1)   # (N*k, 1)
        
        # Instead of expanding x, we use indices to find which tokens go where
        # This is the "Gather" step
        for i in range(self.num_experts):
            # Find which slots in the flat_indices want this expert
            # 'condition.nonzero()' is faster than boolean masking on CPU
            batch_indices = (flat_indices == i).nonzero(as_tuple=True)[0]
            
            if batch_indices.numel() == 0:
                continue
                
            # Map back to original token IDs (batch_indices // top_k)
            source_token_ids = batch_indices // self.top_k
            
            # Efficiently pull only the needed tokens
            expert_in = x_flat.index_select(0, source_token_ids)
            
            # Manual Expert Forward (Fused ReLU^2)
            # Using the sequential layers directly to avoid Module overhead
            curr_expert = self.experts[i]
            mid = curr_expert[0](expert_in)
            mid = torch.pow(mid.clamp(min=0), 2) # Fast ReLU^2
            expert_out = curr_expert[2](mid)
            
            # Weighted Scatter-Add
            # final_output[source_tokens] += expert_out * weight
            actual_weights = flat_weights.index_select(0, batch_indices)
            final_output.index_add_(0, source_token_ids, expert_out * actual_weights)

        return final_output.view(B, T, C)


class EfficientMoE(nn.Module):
    """
    Efficient Mixture of Experts module with sparse top-k routing.
    Features:
    - Multiple expert MLPs
    - Top-k routing (typically k=1 or k=2)
    - Load balancing loss to ensure experts are used evenly
    - Follows codebase conventions: no bias, relu^2 activation
    """
    def __init__(self, config, num_experts=8, top_k=2, load_balance_weight=0.01, d_expert=None, checkpointing=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.checkpointing = checkpointing
        self.n_embd = config.n_embd
        
        # Default d_expert to 4 * n_embd (same as standard MLP)
        if d_expert is None:
            d_expert = 4 * config.n_embd * top_k // num_experts
        self.d_expert = d_expert
        
        # Router: projects input to logits for each expert
        self.router = nn.Linear(config.n_embd, num_experts, bias=False)
        
        # Create multiple expert MLPs
        self.experts = nn.ModuleList([
            Expert(config.n_embd, d_expert) for _ in range(num_experts)
        ])
        
        # Track load balancing statistics during forward pass
        self._load_balance_loss = None

    def forward(self, x):
        """
        Forward pass through MoE layer.
        Args:
            x: Input tensor of shape (B, T, n_embd)
        Returns:
            output: Output tensor of shape (B, T, n_embd)
        """
        B, T, C = x.shape
        original_shape = x.shape
        
        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, C)  # (B*T, n_embd)
        
        # Compute router logits
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        
        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (B*T, top_k)
        
        # Compute routing probabilities (softmax over top-k)
        router_probs = F.softmax(top_k_logits, dim=-1)  # (B*T, top_k)
        
        # Initialize output tensor
        output = torch.zeros_like(x_flat)  # (B*T, n_embd)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that route to this expert
            expert_mask = (top_k_indices == expert_idx)  # (B*T, top_k)
            expert_mask_any = expert_mask.any(dim=-1)  # (B*T,)
            
            if not expert_mask_any.any():
                continue  # No tokens route to this expert
            
            # Get the weights for this expert (sum over top_k if token routes to expert multiple times)
            expert_weights = (router_probs * expert_mask.float()).sum(dim=-1)  # (B*T,)
            expert_weights = expert_weights * expert_mask_any.float()  # Zero out non-routed tokens
            
            # Get tokens that route to this expert
            expert_input = x_flat[expert_mask_any]  # (num_tokens, n_embd)
            
            # Forward through expert
            if self.training and self.checkpointing and expert_input.requires_grad:
                expert_output = checkpoint(self.experts[expert_idx], expert_input, use_reentrant=False)
            else:
                expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, n_embd)
            
            # Accumulate weighted outputs
            output[expert_mask_any] += expert_output * expert_weights[expert_mask_any].unsqueeze(-1)
        
        # Reshape back to original shape
        output = output.view(original_shape)  # (B, T, n_embd)
        
        # Compute load balancing loss
        if self.training:
            # Compute fraction of tokens routed to each expert
            router_probs_all = F.softmax(router_logits, dim=-1)  # (B*T, num_experts)
            # Average over tokens to get expert usage
            expert_usage = router_probs_all.mean(dim=0)  # (num_experts,)
            # Load balancing loss: encourage uniform distribution
            # Using coefficient of variation squared as loss
            uniform_target = 1.0 / self.num_experts
            load_balance_loss = self.load_balance_weight * self.num_experts * ((expert_usage - uniform_target) ** 2).sum()
            self._load_balance_loss = load_balance_loss
        else:
            self._load_balance_loss = None
        
        return output

    def get_load_balance_loss(self):
        """Get the load balancing loss from the last forward pass."""
        return self._load_balance_loss if self._load_balance_loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)


class ScatterMoE(nn.Module):
    """
    ScatterMoE wrapper that uses the Triton-based ScatterMoE implementation.
    This provides efficient GPU inference and training with reduced memory usage.
    
    Features:
    - Uses scattermoe's ParallelLinear for fused expert operations
    - Top-k routing (typically k=1 or k=2)
    - Follows codebase conventions: no bias, relu^2 activation
    """
    def __init__(self, config, num_experts=8, top_k=2, load_balance_weight=0.01, d_expert=None):
        super().__init__()
        if not SCATTERMOE_AVAILABLE:
            raise ImportError(
                "scattermoe is not installed. Install it with:\n"
                "  pip install git+https://github.com/shawntan/scattermoe.git\n"
                "Or use the 'gpu' or 'rocm' extra: pip install nanochat[gpu]"
            )
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.n_embd = config.n_embd
        
        # Default d_expert to 4 * n_embd (same as standard MLP)
        if d_expert is None:
            d_expert = 4 * config.n_embd
        self.d_expert = d_expert
        
        # Router: projects input to logits for each expert
        self.router = nn.Linear(config.n_embd, num_experts, bias=False)
        
        # Create ScatterMoE MLP module
        # ScatterMoE uses a custom activation, but we need relu^2
        # We'll create a custom activation function for relu^2
        class ReLU2(nn.Module):
            def forward(self, x):
                return F.relu(x).square()
        
        # Initialize ScatterMoE MLP with relu^2 activation
        # Try with bias=False first, fall back if not supported
        try:
            self.mlp = ScatterMoE_MLP(
                input_size=config.n_embd,
                hidden_size=d_expert,
                activation=ReLU2(),
                num_experts=num_experts,
                top_k=top_k,
                bias=False,  # Follow codebase convention: no bias
            )
        except TypeError:
            # If bias parameter is not supported, try without it
            # ScatterMoE may handle bias internally
            self.mlp = ScatterMoE_MLP(
                input_size=config.n_embd,
                hidden_size=d_expert,
                activation=ReLU2(),
                num_experts=num_experts,
                top_k=top_k,
            )
        
        # Track load balancing statistics during forward pass
        self._load_balance_loss = None

    def forward(self, x):
        """
        Forward pass through ScatterMoE layer.
        Args:
            x: Input tensor of shape (B, T, n_embd)
        Returns:
            output: Output tensor of shape (B, T, n_embd)
        """
        B, T, C = x.shape
        original_shape = x.shape
        
        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, C)  # (B*T, n_embd)
        
        # Compute router logits
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        
        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (B*T, top_k)
        
        # Compute routing probabilities (softmax over top-k)
        router_probs = F.softmax(top_k_logits, dim=-1)  # (B*T, top_k)
        
        # ScatterMoE expects k_weights and k_idxs
        # k_weights: (B*T, top_k) - routing weights
        # k_idxs: (B*T, top_k) - expert indices
        k_weights = router_probs
        k_idxs = top_k_indices
        
        # Forward through ScatterMoE MLP
        output = self.mlp(x_flat, k_weights, k_idxs)  # (B*T, n_embd)
        
        # Reshape back to original shape
        output = output.view(original_shape)  # (B, T, n_embd)
        
        # Load balancing loss
        if self.training:
            # Compute fraction of tokens routed to each expert
            router_probs_all = F.softmax(router_logits, dim=-1)  # (B*T, num_experts)
            # Average over tokens to get expert usage
            expert_usage = router_probs_all.mean(dim=0)  # (num_experts,)
            # Load balancing loss: encourage uniform distribution
            uniform_target = 1.0 / self.num_experts
            load_balance_loss = self.load_balance_weight * self.num_experts * ((expert_usage - uniform_target) ** 2).sum()
            self._load_balance_loss = load_balance_loss
        else:
            self._load_balance_loss = None
        
        return output

    def get_load_balance_loss(self):
        """Get the load balancing loss from the last forward pass."""
        return self._load_balance_loss if self._load_balance_loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)


class MoE(nn.Module):
    """
    Main MoE entry point.
    Automatically selects the best backend (ScatterMoE if available and on GPU, else EfficientMoE).
    """
    def __init__(self, config, num_experts=8, top_k=2, load_balance_weight=0.01, d_expert=None, checkpointing=False, backend='auto'):
        super().__init__()
        
        # Backend selection logic
        if backend == 'auto':
            self.use_scattermoe = SCATTERMOE_AVAILABLE
        elif backend == 'scattermoe':
            if not SCATTERMOE_AVAILABLE:
                raise ImportError("ScatterMoE backend requested but scattermoe is not installed.")
            self.use_scattermoe = True
        elif backend == 'efficient':
            self.use_scattermoe = False
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'auto', 'scattermoe', or 'efficient'.")
        
        if self.use_scattermoe:
            # ScatterMoE usually handles memory efficiently, checkpointing might not be needed or supported the same way
            self.impl = ScatterMoE(config, num_experts, top_k, load_balance_weight, d_expert)
        else:
            self.impl = EfficientMoE(config, num_experts, top_k, load_balance_weight, d_expert, checkpointing=checkpointing)

    def forward(self, x):
        return self.impl(x)

    def get_load_balance_loss(self):
        return self.impl.get_load_balance_loss()
