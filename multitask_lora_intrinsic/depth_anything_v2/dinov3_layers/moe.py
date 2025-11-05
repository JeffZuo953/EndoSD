import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from .ffn_layers import Mlp_LoRA


class Expert(nn.Module):
    """An expert module, which is a simple MLP with LoRA applied to its linear layers."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 1,
    ):
        super().__init__()
        self.mlp = Mlp_LoRA(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop,
            bias=bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Re-initialize weights to smaller values to prevent instability
        for param in self.mlp.parameters():
            if param.dim() > 1: # Only re-initialize weight matrices, not biases
                nn.init.normal_(param, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TopKRouter(nn.Module):
    """Router module that selects the top-k experts for each token."""
    def __init__(self, in_features: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(in_features, num_experts)

    def forward(self, x: torch.Tensor):
        logits = self.linear(x)
        gates = F.softmax(logits, dim=-1)
        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1, largest=True, sorted=False)
        return top_k_gates, top_k_indices


class MoELayer(nn.Module):
    """Mixture of Experts layer."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(in_features, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(
                in_features, hidden_features, out_features, act_layer, drop, bias, lora_rank, lora_alpha
            ) for _ in range(num_experts)
        ])
        
        self.aux_loss_coef = 0.01 # Coefficient for the auxiliary load balancing loss

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, in_features)
        batch_size, seq_len, in_features = x.shape
        x_flat = x.view(-1, in_features) # (batch_size * seq_len, in_features)

        gates, indices = self.router(x_flat) # gates: (N, top_k), indices: (N, top_k) where N = batch_size * seq_len
        
        # Auxiliary load balancing loss
        # This encourages the router to distribute tokens evenly across experts.
        gates_flat = gates.view(-1)
        expert_counts = x.new_zeros(self.num_experts)
        expert_counts.index_add_(0, indices.view(-1), torch.ones_like(gates_flat))
        total_tokens = expert_counts.sum().float()
        if total_tokens > 0:
            load_balance_loss = (expert_counts * F.softmax(expert_counts, dim=0)).sum() / total_tokens
        else:
            # ensure aux loss stays on the same device/dtype as inputs
            load_balance_loss = x.new_tensor(0.0)
        self.aux_loss = self.aux_loss_coef * load_balance_loss

        # Dispatch tokens to experts
        final_output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            expert_mask = (indices == i).any(dim=-1)
            if expert_mask.any():
                tokens_for_expert = x_flat[expert_mask]
                gate_values = gates[expert_mask, (indices[expert_mask] == i).nonzero(as_tuple=True)[1]]
                
                expert_output = self.experts[i](tokens_for_expert)
                
                # Use non-inplace operation
                final_output = final_output.scatter_add(0, expert_mask.nonzero(as_tuple=True)[0].unsqueeze(-1).expand_as(tokens_for_expert), expert_output * gate_values.unsqueeze(-1))
        
        output = final_output
        
        return output.view(batch_size, seq_len, -1)
