import torch
import torch.nn as nn

class Segmenter(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def boundaries(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.linear(features).squeeze(-1)
        return torch.sigmoid(logits)

    def pool(self, features: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return features[start:end].mean(dim=0)

    def segment(self, features: torch.Tensor, threshold: float = 0.5) -> list[tuple[int, int]]:
        probs = self.boundaries(features)
        indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
        segments = []
        prev = 0
        for idx in indices:
            segments.append((prev, idx + 1))
            prev = idx + 1
        if prev < len(features):
            segments.append((prev, len(features)))
        return segments
