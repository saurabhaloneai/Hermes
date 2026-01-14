import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_experts
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        gate_up = torch.einsum("th,ehq->teq", x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_outputs = torch.einsum("teq,eqh->teh", F.silu(gate) * up, self.down_proj)
        weighted = expert_outputs * routing_weights.unsqueeze(-1)
        return weighted.sum(dim=1)


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp
        self.num_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = MoEExperts(config)

    def forward(self, x):
        B, T, _ = x.shape
        hidden = x.reshape(-1, self.hidden_size)

        router_logits = self.gate(hidden)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        routed = torch.zeros_like(router_logits)
        topk_weights = topk_weights.to(router_logits.dtype)
        routed.scatter_(1, topk_indices, topk_weights)
        routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-9)
        routed = routed.to(hidden.dtype)
        expert_out = self.experts(hidden, routed)
        combined = expert_out.view(B, T, self.hidden_size)

        return combined
