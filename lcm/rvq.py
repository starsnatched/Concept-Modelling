import torch
import torch.nn as nn

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, levels: int):
        super().__init__()
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, embedding_dim) for _ in range(levels)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        codes = []
        quantized = torch.zeros_like(x)
        for cb in self.codebooks:
            distances = (residual.unsqueeze(1) - cb.weight.unsqueeze(0)).pow(2).sum(dim=-1)
            idx = distances.argmin(dim=1)
            codes.append(idx)
            quant = cb(idx)
            quantized = quantized + quant
            residual = residual - quant
        return quantized, torch.stack(codes, dim=1)
