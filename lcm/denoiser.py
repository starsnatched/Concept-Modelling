import torch
import torch.nn as nn

class ConceptDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 512, latent_len: int = 128, heads: int = 8):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(latent_len, latent_dim))
        self.self_attn = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.cross_local = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.cross_retrieved = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, local: torch.Tensor, retrieved: torch.Tensor | None, steps: int) -> torch.Tensor:
        z = self.latent.unsqueeze(0).repeat(local.size(0), 1, 1)
        for _ in range(steps):
            z, _ = self.self_attn(z, z, z)
            z, _ = self.cross_local(z, local, local)
            if retrieved is not None:
                z, _ = self.cross_retrieved(z, retrieved, retrieved)
            z = self.linear(z)
        return z
