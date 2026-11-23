"""F2 candidate: causal fingertip-force encoder + zero-init Force-gated FiLM on action tokens.
At initialization gamma=beta=0 so the module is the exact identity."""
import torch
import torch.nn as nn


class ForceFiLM(nn.Module):
    def __init__(self, force_dim=32, hidden_dim=1024, mid_dim=256):
        super().__init__()
        self.force_dim = int(force_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.force_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 2 * self.hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, force_feat):
        gb = self.net(force_feat)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta

    @staticmethod
    def modulate(tokens, gamma, beta):
        # tokens [B,T,H]; gamma/beta [B,H]
        return tokens * (1.0 + gamma[:, None, :].to(tokens.dtype)) + beta[:, None, :].to(tokens.dtype)