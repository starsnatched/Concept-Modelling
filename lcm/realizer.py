import torch
import torch.nn as nn

class ByteDiffusionDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, steps: int = 2):
        super().__init__()
        self.steps = steps
        self.layers = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(steps)])
        self.head = nn.Linear(latent_dim, 256)
    def forward(self, plan: torch.Tensor) -> torch.Tensor:
        x = plan
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits
    def sample(self, plan: torch.Tensor) -> bytes:
        logits = self.forward(plan)
        bytes_tensor = logits.argmax(dim=-1).to(torch.uint8)
        return bytes_tensor.flatten().cpu().numpy().tobytes()

class SegmentCTCRealizer(nn.Module):
    def __init__(self, latent_dim: int = 512, max_len: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.length_predictor = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, 256)
    def decode(self, plan: torch.Tensor) -> bytes:
        lengths = self.length_predictor(plan).relu().floor().clamp(min=1, max=self.max_len).long()
        logits = self.classifier(plan)
        codes = logits.argmax(dim=-1).to(torch.uint8)
        output = []
        for i in range(plan.size(1)):
            output.extend([codes[0, i].item()] * lengths[0, i].item())
        return bytes(output)
