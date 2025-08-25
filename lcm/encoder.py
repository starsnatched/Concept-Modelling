import torch
import torch.nn as nn

class StreamingEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(256, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, data: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(data)
        out, state = self.gru(x, state)
        return out, state

    def step(self, byte: int, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([[byte]], dtype=torch.long)
        out, state = self.forward(x, state)
        return out[:, -1], state
