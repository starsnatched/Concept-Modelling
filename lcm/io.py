import torch
from pathlib import Path
from .encoder import StreamingEncoder
from .segmenter import Segmenter
from .rvq import ResidualVectorQuantizer
from .utils import get_device


def save_models(path: str, encoder: StreamingEncoder, segmenter: Segmenter, rvq: ResidualVectorQuantizer) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), p / "encoder.pt")
    torch.save(segmenter.state_dict(), p / "segmenter.pt")
    torch.save(rvq.state_dict(), p / "rvq.pt")


def load_models(path: str, device: torch.device | None = None) -> tuple[StreamingEncoder, Segmenter, ResidualVectorQuantizer]:
    device = device or get_device()
    encoder = StreamingEncoder().to(device)
    segmenter = Segmenter().to(device)
    rvq = ResidualVectorQuantizer(1024, 512, 2).to(device)
    p = Path(path)
    encoder.load_state_dict(torch.load(p / "encoder.pt", map_location=device))
    segmenter.load_state_dict(torch.load(p / "segmenter.pt", map_location=device))
    rvq.load_state_dict(torch.load(p / "rvq.pt", map_location=device))
    return encoder, segmenter, rvq
