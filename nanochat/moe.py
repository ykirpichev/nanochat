import torch
import torch.nn as nn
import primus_turbo.pytorch as turbo # or from aiter import ops as pt_ops

class PrimusTurboLocalMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, intermediate_size, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        
        # 1. Router (Standard Linear layer for gating)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 2. Expert Weights (Packed for Turbo Grouped GEMM)
        # We store them as a single parameter to use the fused kernel
        # Shape: [num_experts, hidden_dim, inter_dim]
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        
        # Initialize weights (Simplified)
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        # x shape: [B * T, Hidden]
        
        # STEP 1: Fused Routing
        # This returns weights, indices, and the aux_loss (for load balancing)
        topk_weights, topk_indices, aux_loss = turbo.ops.moe.fused_group_topk_routing_with_aux_score(
            self.gate(x), 
            top_k=self.top_k,
            aux_loss_weight=0.01
        )

        # STEP 2: Local Token Permutation
        # Rearranges x so tokens for the same expert are contiguous.
        # Returns: permuted tensor, row_id_map (to undo), and counts per expert.
        permuted_x, row_id_map, tokens_per_expert = turbo.ops.moe.token_permute(
            x, 
            topk_indices
        )

        # STEP 3: Fused Expert Execution (Turbo Grouped MLP)
        # This avoids launching 'num_experts' separate kernels.
        # It processes all experts in one pass over the permuted data.
        expert_out = turbo.ops.moe.use_turbo_groupmlp(
            permuted_x, 
            self.w1, 
            self.w2, 
            tokens_per_expert
        )

        # STEP 4: Local Token Un-permutation
        # Re-orders expert_out back to the original sequence and scales by gate weights.
        output = turbo.ops.moe.token_unpermute(
            expert_out, 
            row_id_map, 
            topk_weights
        )

        return output, aux_loss


if __name__ == "__main__":
    model = PrimusTurboLocalMoELayer(hidden_size=1024, num_experts=16, intermediate_size=4096)
    x = torch.randn(1, 1024)
    output, aux_loss = model(x)
    print(output.shape)
    print(aux_loss)