import torch
import torch.nn as nn

class ConceptPlanner(nn.Module):
    def __init__(self, latent_dim: int = 512, latent_len: int = 128, heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_len = latent_len
        self.self_attn = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.cross_local = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.cross_retrieved = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.linear = nn.Linear(latent_dim, latent_dim)
    def forward(self, local: torch.Tensor, retrieved: torch.Tensor | None, steps: int) -> torch.Tensor:
        batch = local.size(0)
        z = torch.randn(batch, self.latent_len, self.latent_dim, device=local.device)
        for _ in range(steps):
            z, _ = self.self_attn(z, z, z)
            z, _ = self.cross_local(z, local, local)
            if retrieved is not None:
                z, _ = self.cross_retrieved(z, retrieved, retrieved)
            z = self.linear(z)
        return z
